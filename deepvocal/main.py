from contextlib import asynccontextmanager
from hmac import compare_digest
import json
import logging
import os
from pathlib import Path
from threading import Event, Thread
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import audio
from .corpus import DATA_DIR
from .credentials import CredentialError, transform
from .model import Model, ModelError
from .retrieval import build_retriever
from .skills import LazyRegistry
from .storage import Storage
from .workflow import assessment_summary, build_workflow, make_plan


ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
Provider = Literal["openai", "deepseek", "dashscope", "custom"]
Melody = Literal["warmup", "triad"]
Focus = Literal["general", "pitch_flat", "pitch_sharp", "timing", "stability", "register"]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelInput(Input):
    provider: Provider = "deepseek"
    model: str = Field(min_length=1, max_length=120)
    use_saved_model: bool = False


class SavedModelInput(Input):
    provider: Provider | Literal["local"]
    model: str | None = Field(default=None, min_length=1, max_length=120)


class ChatInput(Input):
    query: str = Field(min_length=1, max_length=2400)
    session_id: Identifier | None = None
    intent: Literal["auto", "qa", "assessment", "practice_plan", "singing"] = "auto"
    retrieval: Literal["vector", "rerank", "graph"] = "graph"
    provider: Provider | None = None
    model: str | None = Field(default=None, min_length=1, max_length=120)
    use_saved_model: bool = False
    audio_id: Identifier | None = None
    assessment_job_id: Identifier | None = None
    melody_id: Melody = "warmup"
    bpm: int = Field(default=90, ge=50, le=160)
    transpose: int = Field(default=0, ge=-12, le=12)
    offset_seconds: float = Field(default=0, ge=0, le=30)
    days: int | None = Field(default=None, ge=1, le=14)
    minutes: int | None = Field(default=None, ge=5, le=90)
    focus: Focus | None = None
    goal: str | None = Field(default=None, min_length=1, max_length=240)
    lyrics: str | None = Field(default=None, min_length=1, max_length=1000)
    voice: Identifier = "default"


class ProgressInput(Input):
    melody_id: Melody
    completed_days: int = Field(ge=0, le=365)
    note: str = Field(default="", max_length=400)


class SkillInput(Input):
    focus: Focus


class ReviewInput(Input):
    approve: bool


def choose_model(provider, model, key, saved=None):
    if saved is not None:
        if provider is not None and provider != saved["provider"]:
            raise HTTPException(409, "已保存的服务商发生变化，请重新打开模型设置。")
        provider = saved["provider"]
        model = model or saved["model"]
        if not key and saved["protected_key"]:
            try:
                key = transform(saved["protected_key"], protect=False).decode("ascii")
            except UnicodeDecodeError:
                raise CredentialError("已保存的 API Key 无法读取，请重新填写并保存。") from None
    configured_provider = os.getenv("DEEPVOCAL_API_PROVIDER", "deepseek")
    provider = provider or configured_provider
    if provider == configured_provider:
        key = key or os.getenv("DEEPVOCAL_API_KEY", "")
        model = model or os.getenv("DEEPVOCAL_CHAT_MODEL")
    if not (key or (provider == "custom" and model)):
        return None
    if not model:
        raise HTTPException(422, "填写模型名称，或在服务端配置 DEEPVOCAL_CHAT_MODEL。")
    return Model(provider, model, key or "")


def requested_model(store, body, key):
    saved = store.model_settings() if body.use_saved_model else None
    if body.use_saved_model and saved is None and not key:
        raise HTTPException(409, "没有可用的已保存设置，请重新打开模型设置。")
    return choose_model(body.provider, body.model, key, saved)


def public_model_settings(store):
    row = store.model_settings()
    if row is None:
        return {"provider": "local", "model": "", "key_saved": False}
    return {"provider": row["provider"], "model": row["model"], "key_saved": bool(row["protected_key"])}


def run_jobs(store, skills, stop):
    while not stop.is_set():
        row = store.claim()
        if row is None:
            stop.wait(0.15)
            continue
        payload = json.loads(row["payload"])
        try:
            if row["kind"] == "memory":
                store.store_fact(payload)
                result = {}
            elif row["kind"] == "assessment":
                result = audio.assess((store.media / f"{payload['audio_id']}.wav").read_bytes(), payload)
                result["summary"] = assessment_summary(result)
                if "plan_request" in payload:
                    result["plan"] = make_plan(store, skills, payload["plan_request"], result)
            elif row["kind"] == "singing":
                wav, kind = audio.generate(payload)
                key = store.put_audio(row["id"], f"{kind}.wav", kind, wav)
                result = {"type": kind, "audio_url": f"/api/audio/{key}"}
            else:
                raise ValueError("未知任务类型。")
            store.finish(row, result)
        except ValueError as exc:
            store.finish(row, error=str(exc))
        except Exception:
            log.exception("Job failed: %s", row["id"])
            store.finish(row, error="任务未完成。请检查录音或服务配置。")


def create_app(data_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        directory = data_dir or os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR
        app.state.store = store = Storage(directory)
        app.state.skills = skills = LazyRegistry(store)
        app.state.retriever = retriever = build_retriever(store)
        app.state.workflow = build_workflow(store, retriever, skills)
        store.recover()
        stop = Event()
        worker = Thread(target=run_jobs, args=(store, skills, stop), daemon=True, name="deepvocal-worker")
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
            if retriever.client:
                retriever.client.close()

    app = FastAPI(title="DeepVocal 声乐导师", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.middleware("http")
    async def require_access_token(request, call_next):
        token = os.getenv("DEEPVOCAL_ACCESS_TOKEN")
        supplied = request.headers.get("x-deepvocal-token", "")
        if token and request.url.path.startswith("/api/") and not compare_digest(token.encode(), supplied.encode()):
            return JSONResponse({"detail": "请提供有效的 X-DeepVocal-Token。"}, status_code=401)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ModelError)
    async def model_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502, headers={"Cache-Control": "no-store"})

    @app.exception_handler(CredentialError)
    async def credential_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        retriever = request.app.state.retriever
        return {"status": "ok", "retrieval_backend": retriever.mode,
                "embedding_model": retriever.embedding_id,
                "embedding_dimensions": retriever.dimensions if retriever.embed else None,
                "knowledge_chunks": len(retriever.rows),
                "imported_documents": retriever.imported_documents,
                "singing_configured": bool(os.getenv("DEEPVOCAL_SINGING_URL"))}

    @app.get("/api/model/settings")
    def get_model_settings(request: Request):
        return public_model_settings(request.app.state.store)

    @app.put("/api/model/settings")
    def save_model_settings(body: SavedModelInput, request: Request,
                            key: Annotated[str | None, Header(alias="X-Model-API-Key")] = None):
        store = request.app.state.store
        key = (key or "").strip() if body.provider != "local" else ""
        if body.provider != "local" and not body.model:
            raise HTTPException(422, "请填写模型 ID。")
        if key and (len(key) > 4096 or not all(33 <= ord(char) <= 126 for char in key)):
            raise HTTPException(422, "API Key 格式无效。")
        protected_key = transform(key.encode("ascii"), protect=True) if key else None
        try:
            store.update_model_settings(body.provider, body.model, protected_key)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return public_model_settings(store)

    @app.post("/api/model/test")
    def test_model(body: ModelInput, request: Request,
                   key: Annotated[str | None, Header(alias="X-Model-API-Key")] = None):
        client = requested_model(request.app.state.store, body, key)
        if client is None:
            raise HTTPException(422, "请填写模型 API Key。")
        client.chat([{"role": "user", "content": "Reply with the single word OK."}])
        return {"status": "ok", "provider": body.provider, "model": body.model}

    @app.get("/api/melodies")
    def melodies():
        return [{"id": key, **value} for key, value in audio.MELODIES.items()]

    @app.get("/api/melodies/{melody_id}/guide.wav")
    def guide(melody_id: Melody, bpm: int = 90, transpose: int = 0):
        if not 50 <= bpm <= 160 or not -12 <= transpose <= 12:
            raise HTTPException(422, "速度需在 50–160 BPM，移调需在 -12 至 12 个半音。")
        wav = audio.guide(audio.MELODIES[melody_id]["notes"], bpm, transpose)
        return Response(wav, media_type="audio/wav")

    @app.post("/api/audio", status_code=201)
    def upload(file: UploadFile, request: Request):
        data = file.file.read(audio.MAX_BYTES + 1)
        if len(data) > audio.MAX_BYTES:
            raise HTTPException(413, "录音大小最多 12 MiB。")
        try:
            audio.read_wave(data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        key = uuid4().hex
        request.app.state.store.put_audio(key, (file.filename or "recording.wav")[:180], "input", data)
        return {"id": key, "audio_url": f"/api/audio/{key}"}

    @app.get("/api/audio/{audio_id}")
    def media(audio_id: str, request: Request):
        store = request.app.state.store
        entry = store.audio(audio_id)
        if not entry:
            raise HTTPException(404, "找不到该音频。")
        return FileResponse(store.media / f"{entry['id']}.wav", media_type="audio/wav")

    @app.post("/api/chat")
    def chat(body: ChatInput, request: Request,
             key: Annotated[str | None, Header(alias="X-Model-API-Key")] = None):
        store = request.app.state.store
        session = body.session_id or uuid4().hex
        if body.session_id and store.session_deleted(session):
            raise HTTPException(410, "该会话已删除，请在新会话中重新发送。")
        client = requested_model(store, body, key)
        result = request.app.state.workflow.invoke({
            "request": body.model_dump(), "history": store.history(session), "model": client,
        })
        result = {k: result[k] for k in ("steps", "answer", "evidence", "assessment", "plan", "job") if k in result}
        profile = {k: v for k in ("goal", "minutes") if (v := getattr(body, k)) is not None}
        turn = store.record_turn(session, body.query, result, profile)
        if turn is None:
            raise HTTPException(410, "该会话已删除，请在新会话中重新发送。")
        return {"session_id": session, "turn_id": turn, **result}

    @app.get("/api/sessions")
    def sessions(request: Request, limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)):
        return request.app.state.store.sessions(limit, offset)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: Identifier, request: Request):
        request.app.state.store.delete_session(session_id)
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/turns")
    def session_turns(session_id: Identifier, request: Request, limit: int = Query(30, ge=1, le=100),
                      before: int | None = Query(None, ge=1)):
        result = request.app.state.store.session_turns(session_id, limit, before)
        if result is None:
            raise HTTPException(404, "找不到该会话。")
        return result

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        result = request.app.state.store.job(job_id)
        if result is None:
            raise HTTPException(404, "找不到该任务。")
        return result

    @app.get("/api/memory")
    def memory(request: Request):
        return {"facts": request.app.state.store.facts()}

    @app.post("/api/memory/progress", status_code=202)
    def progress(body: ProgressInput, request: Request):
        return {"event_id": request.app.state.store.record_progress(body.model_dump())}

    @app.get("/api/skills")
    def list_skills(request: Request):
        return request.app.state.skills.list()

    @app.post("/api/skills/candidates")
    def propose(body: SkillInput, request: Request):
        try:
            return request.app.state.skills.propose(body.focus)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/skills/{name}/{version}")
    def skill(name: str, version: int, request: Request):
        result = request.app.state.skills.draft(name, version)
        if result is None:
            raise HTTPException(404, "找不到该 Skill 版本。")
        return result

    @app.post("/api/skills/{name}/{version}/review")
    def review(name: str, version: int, body: ReviewInput, request: Request):
        try:
            return request.app.state.skills.review(name, version, body.approve)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    return app


app = create_app()
