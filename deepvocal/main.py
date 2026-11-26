from contextlib import aclosing, asynccontextmanager
from hmac import compare_digest
import json
import logging
import os
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from . import audio
from .assessment import build_assessment_graph, index_song_reference
from .assessment_agent import build_react_agent
from .corpus import ChunkId, DATA_DIR
from .credentials import CredentialError, transform
from .memory import MemoryService
from .model import Model, ModelError
from .retrieval import LOCAL_MODEL, build_retriever
from .skills import LazyRegistry
from .singing_agent import COVER_CUE, reply_title, requested_title
from .storage import Storage
from .voice_conversion import SOURCE_SECONDS, VoiceConversionError, convert_singing_voice, convert_song, service_url
from .workflow import (
    ASSESSMENT_REQUEST, KNOWLEDGE_REQUEST, PLAN_REQUEST, Outcome, Task,
    assessment_summary, build_workflow, resume_assessment,
)


ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
Provider = Literal["openai", "deepseek", "dashscope", "custom"]
Melody = Literal["warmup", "triad"]
Focus = Literal["general", "pitch_flat", "pitch_sharp", "timing", "stability", "register"]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")]
SongTitle = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=180, pattern=r"[\p{L}\p{N}]")]
CHAT_RESULT_FIELDS = ("steps", "answer", "evidence", "related_figures", "qa_generation", "rag",
                      "assessment", "plan", "planning_agent", "singing_agent", "job", "jobs", "task_statuses")


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelInput(Input):
    provider: Provider = "deepseek"
    model: str = Field(min_length=1, max_length=120)
    use_saved_model: bool = False


class SavedModelInput(Input):
    provider: Provider | Literal["local"]
    model: str | None = Field(default=None, min_length=1, max_length=120)


class VoiceConversionInput(Input):
    source_vocal_audio_id: Identifier = Field(description="想模仿的参考主唱片段")
    timbre_audio_id: Identifier = Field(description="用户本人的音色录音")
    semitone_shift: int = Field(default=0, ge=-12, le=12, strict=True)


class ChatInput(Input):
    query: str = Field(min_length=1, max_length=2400)
    session_id: Identifier | None = None
    intent: Literal["auto", "qa", "assessment", "practice_plan", "singing", "qa_assessment"] = "auto"
    retrieval: Literal["vector", "rerank", "graph", "cross_rerank", "cross_graph"] = "graph"
    provider: Provider | None = None
    model: str | None = Field(default=None, min_length=1, max_length=120)
    use_saved_model: bool = False
    audio_id: Identifier | None = None
    song_title: SongTitle | None = None
    alignment_mode: Literal["llm", "dtw", "signal"] = "llm"
    reference_mode: Literal["library", "builtin"] = "library"
    reference_audio_id: Identifier | None = None
    reference_start_seconds: float | None = Field(default=None, ge=0, le=audio.MAX_REFERENCE_SECONDS)
    reference_end_seconds: float | None = Field(default=None, gt=0, le=audio.MAX_REFERENCE_SECONDS)
    assessment_job_id: Identifier | None = None
    assessment_source: Literal["session", "explicit"] | None = None
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
    singing_mode: Literal["cover", "guide"] | None = None

    @model_validator(mode="after")
    def validate_reference_range(self):
        start, end = self.reference_start_seconds, self.reference_end_seconds
        if start is None and end is None:
            return self
        if not self.reference_audio_id or start is None or end is None:
            raise ValueError("参考片段需要参考音频、起点和结束时间。")
        if not 0.5 <= end - start <= 90:
            raise ValueError("手动选择的参考片段需为 0.5–90 秒。")
        return self


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


def continue_cover(store, body):
    request = body.model_dump()
    if not body.session_id or body.intent not in ("auto", "singing") or body.singing_mode == "guide":
        return request
    pending = store.pending_cover(body.session_id)
    if not pending:
        return request
    query = body.query
    if any(cue.search(query) for cue in (ASSESSMENT_REQUEST, KNOWLEDGE_REQUEST, PLAN_REQUEST)):
        return request
    title = requested_title(request)
    if pending["issue"] == "song_title_required" and query not in {"谢谢", "不用了", "你好", "取消"}:
        title = title or reply_title(query)
    is_cover = body.intent == "singing" or body.singing_mode == "cover" or COVER_CUE.search(query)
    if title or body.reference_audio_id or (body.audio_id and is_cover):
        request.update(singing_mode="cover", audio_id=body.audio_id or pending.get("recording_audio_id"),
                       song_title=title or pending.get("song_title"))
    return request


def assessment_model_for(store, body, key, client):
    recovery = "request" if key else "saved" if body.use_saved_model else "environment"
    if not client and body.provider is None and body.model is None and not key:
        if saved := store.model_settings():
            return choose_model(None, None, None, saved), "saved"
    return client, recovery


def public_model_settings(store):
    row = store.model_settings()
    if row is None:
        return {"provider": "local", "model": "", "key_saved": False}
    return {"provider": row["provider"], "model": row["model"], "key_saved": bool(row["protected_key"])}


class AssessmentModels:
    """Keep request credentials in memory and store only recoverable model descriptors."""

    def __init__(self, store):
        self.store = store
        self.pending = {}
        self.lock = Lock()

    def enqueue(self, payload, model, recovery):
        payload["agent_model"] = {"provider": model.provider, "name": model.name, "recovery": recovery}
        key = None
        try:
            with self.store.connect() as db:
                key = self.store.enqueue("assessment", payload, db)
                with self.lock:
                    self.pending[key] = model
        except Exception:
            if key:
                with self.lock:
                    self.pending.pop(key, None)
            raise
        return key

    def take(self, key, config):
        with self.lock:
            pending = self.pending.pop(key, None)
        if pending is not None:
            return pending
        if not config:
            saved = self.store.model_settings()
            return choose_model(None, None, None, saved)
        if config["recovery"] == "request":
            return None
        saved = None
        if config["recovery"] == "saved":
            saved = self.store.model_settings()
            if not saved or saved["provider"] != config["provider"]:
                return None
        return choose_model(config["provider"], config["name"], None, saved)


def finish_assessment_job(store, workflow, agent, models, memory, row, payload):
    state = {}
    model = None
    try:
        model = models.take(row["id"], payload.get("agent_model"))
        if model is None:
            raise ModelError("录音测评需要可调用工具的模型，请配置模型后重新提交。")
        state = agent.invoke({"spec": payload, "model": model})
        status = state["status"]
        error = state.get("error")
    except (CredentialError, ModelError, HTTPException) as exc:
        status, error = "failed", exc.detail if isinstance(exc, HTTPException) else str(exc)
    except Exception:
        log.exception("Assessment agent failed: %s", row["id"])
        status, error = "failed", "测评 Agent 未完成，请重试。"
    trace = {"status": status, "turns": state.get("turns", 0), "actions": state.get("actions", [])}
    report = state.get("report") if status in ("completed", "needs_input") else None
    if report:
        report["summary"] = assessment_summary(report)
        if status == "completed" and report.get("melody_id") and not payload.get("assessment_job_id"):
            report["assessment_settings"] = {key: payload.get(key) for key in ("bpm", "transpose")}
    output = {"assessment": report, "answer": report["summary"]} if report else {
        "answer": error or "测评 Agent 未能提交有效结果。"
    }
    continuation = payload.get("continuation")
    if not continuation and (request := payload.get("plan_request")):
        continuation = {"request": request, "tasks": [
            Task(kind="practice_plan", query=request["query"], depends_on=["assessment"]),
        ]}
    request = continuation["request"] if continuation else {"query": payload.get("assessment_query", "测评")}
    followups = continuation["tasks"] if continuation else []
    merged = resume_assessment(workflow, request,
                               Outcome(status=status, output=output, assessment_job_id=row["id"]), followups, model=model,
                               session_id=continuation.get("session_id") if continuation else None)
    result = {**(report or {}), "assessment_agent": trace, "task_statuses": merged["task_statuses"]}
    if "planning_agent" in merged:
        result["planning_agent"] = merged["planning_agent"]
    if "plan" in merged:
        result["plan"] = merged["plan"]
    elif followups and report:
        result["plan_error"] = "请先获得有效的测评报告，再生成针对本次演唱的计划。" if status == "needs_input" else merged["answer"]
    if report and not payload.get("assessment_job_id"):
        memory.finish_assessment(row, payload, result)
    else:
        store.finish(row, result, error=output["answer"] if status == "failed" else None)


def run_voice_conversion(store, job_id, payload):
    render_song = payload.get("render_song", False)
    output_path = store.media / f"{job_id}.wav"
    if (entry := store.audio(job_id)) and entry["kind"] == "cover" and output_path.is_file():
        wav = output_path.read_bytes()
    else:
        try:
            source = (store.media / f"{payload['source_vocal_audio_id']}.wav").read_bytes()
            timbre = (store.media / f"{payload['timbre_audio_id']}.wav").read_bytes()
        except FileNotFoundError:
            raise VoiceConversionError("参考主唱或本人录音已不可用，请重新上传。") from None
        if render_song:
            def record_progress(done, total):
                store.update_job_progress(job_id, {"segments_done": done, "segments_total": total})

            wav = convert_song(source, timbre, semitone_shift=payload["semitone_shift"], progress=record_progress,
                               start_seconds=payload["reference_start_seconds"], end_seconds=payload["reference_end_seconds"])
        else:
            wav = convert_singing_voice(source, timbre, semitone_shift=payload["semitone_shift"])
        store.put_audio(job_id, "cover.wav", "cover", wav)
    samples, rate = audio.read_wave(wav, audio.MAX_REFERENCE_SECONDS + 1 if render_song else SOURCE_SECONDS + 1)
    result = {"type": "cover", "provider": "seed-vc", "audio_url": f"/api/audio/{job_id}",
              "duration_seconds": round(len(samples) / rate, 3), "sample_rate_hz": rate,
              "semitone_shift": payload["semitone_shift"]}
    if render_song:
        result["song_title"] = payload["song_title"]
    return result


def run_jobs(store, workflow, assessment_agent, models, memory, stop):
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
                finish_assessment_job(store, workflow, assessment_agent, models, memory, row, payload)
                continue
            elif row["kind"] == "singing":
                wav, kind = audio.generate(payload)
                key = store.put_audio(row["id"], f"{kind}.wav", kind, wav)
                result = {"type": kind, "audio_url": f"/api/audio/{key}"}
            elif row["kind"] == "voice_conversion":
                result = run_voice_conversion(store, row["id"], payload)
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
        local_embed = retriever.embed if retriever.embedding_id == LOCAL_MODEL else None
        app.state.memory = memory = MemoryService(store, local_embed)
        memory.backfill_assessments()
        models = AssessmentModels(store)
        app.state.workflow = workflow = build_workflow(store, retriever, skills, memory, models.enqueue)
        app.state.assessment_graph = assessment_graph = build_assessment_graph(store)
        app.state.assessment_agent = assessment_agent = build_react_agent(store, assessment_graph)
        store.recover()
        stop = Event()
        workers = [Thread(target=run_jobs, args=(store, workflow, assessment_agent, models, memory, stop), daemon=True,
                          name=f"deepvocal-worker-{index}") for index in (1, 2)]
        for worker in workers:
            worker.start()
        try:
            yield
        finally:
            stop.set()
            for worker in workers:
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
                "reranker_model": retriever.reranker.name if retriever.reranker else None,
                "knowledge_chunks": len(retriever.rows),
                "imported_documents": retriever.imported_documents,
                "graph": retriever.graph.counts(),
                "singing_configured": bool(os.getenv("DEEPVOCAL_SINGING_URL"))}

    @app.get("/api/knowledge/figures/{chunk_id}/{figure_id}")
    def knowledge_figure(chunk_id: ChunkId, figure_id: ChunkId, request: Request):
        row = request.app.state.retriever.rows.get(chunk_id, {})
        if not any(figure["id"] == figure_id for figure in row.get("figures", [])):
            raise HTTPException(404, "找不到该图版。")
        assets = (request.app.state.store.root / "knowledge-assets").resolve()
        path = (assets / row["document_id"] / f"{figure_id}.png").resolve()
        if not path.is_relative_to(assets) or not path.is_file():
            raise HTTPException(404, "找不到该图版。")
        return FileResponse(path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})

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
            data, _, _ = audio.read_upload(data, file.filename)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        key = uuid4().hex
        request.app.state.store.put_audio(key, (file.filename or "recording.wav")[:180], "input", data)
        return {"id": key, "audio_url": f"/api/audio/{key}"}

    @app.post("/api/audio/reference", status_code=201)
    def upload_reference(file: UploadFile, request: Request, song_title: Annotated[SongTitle | None, Form()] = None):
        data = file.file.read(audio.MAX_REFERENCE_BYTES + 1)
        if len(data) > audio.MAX_REFERENCE_BYTES:
            raise HTTPException(413, "参考音频大小最多 96 MiB。")
        try:
            data, samples, rate = audio.read_upload(data, file.filename, audio.MAX_REFERENCE_SECONDS)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        key = uuid4().hex
        store = request.app.state.store
        store.put_audio(key, (file.filename or "reference.wav")[:180], "reference", data, song_title=song_title)
        index_song_reference(store, key, data, samples, rate)
        return {"id": key, "audio_url": f"/api/audio/{key}", "duration_seconds": len(samples) / rate}

    @app.get("/api/audio/{audio_id}")
    def media(audio_id: str, request: Request):
        store = request.app.state.store
        entry = store.audio(audio_id)
        if not entry:
            raise HTTPException(404, "找不到该音频。")
        return FileResponse(store.media / f"{entry['id']}.wav", media_type="audio/wav")

    @app.post("/api/voice-conversions", status_code=202, summary="提交个人音色翻唱",
              description="将参考主唱和本人录音交给配置的 Seed-VC 服务，默认使用作者的 Hugging Face Space。")
    def voice_conversion(body: VoiceConversionInput, request: Request):
        if body.source_vocal_audio_id == body.timbre_audio_id:
            raise HTTPException(422, "请分别提供参考主唱和本人音色录音。")
        store = request.app.state.store
        for key, label, kinds in (
            (body.source_vocal_audio_id, "参考主唱", ("input", "reference")),
            (body.timbre_audio_id, "本人音色录音", ("input",)),
        ):
            entry = store.audio(key)
            if not entry or not (store.media / f"{key}.wav").is_file():
                raise HTTPException(404, f"找不到{label}。")
            if entry["kind"] not in kinds:
                raise HTTPException(422, f"请通过音频上传接口提供{label}。")
        try:
            service_url()
        except VoiceConversionError as error:
            raise HTTPException(503, str(error)) from None
        job_id = store.enqueue("voice_conversion", body.model_dump())
        return {"job": {"id": job_id, "kind": "voice_conversion"}}

    def prepare_chat(body, request, key):
        store = request.app.state.store
        session = body.session_id or uuid4().hex
        if body.session_id and store.session_deleted(session):
            raise HTTPException(410, "该会话已删除，请在新会话中重新发送。")
        client = requested_model(store, body, key)
        state = {"request": continue_cover(store, body), "history": store.history(session), "model": client,
                 "session_id": session}
        may_assess = body.intent in ("assessment", "qa_assessment") or (
            body.intent == "auto" and (body.audio_id or body.assessment_job_id)
        )
        if may_assess:
            assessment_model, recovery = assessment_model_for(store, body, key, client)
            state.update(assessment_model=assessment_model, assessment_recovery=recovery)
        return store, session, state

    def save_chat(store, session, body, state):
        result = {key: state[key] for key in CHAT_RESULT_FIELDS if key in state}
        profile = {k: v for k in ("goal", "minutes") if (v := getattr(body, k)) is not None}
        turn = store.record_turn(session, body.query, result, profile)
        if turn is None:
            raise HTTPException(410, "该会话已删除，请在新会话中重新发送。")
        return {"session_id": session, "turn_id": turn, **result}

    @app.post("/api/chat")
    def chat(body: ChatInput, request: Request,
             key: Annotated[str | None, Header(alias="X-Model-API-Key")] = None):
        store, session, state = prepare_chat(body, request, key)
        return save_chat(store, session, body, request.app.state.workflow.invoke(state))

    @app.post("/api/chat/stream", response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    def chat_stream(body: ChatInput, request: Request,
                    key: Annotated[str | None, Header(alias="X-Model-API-Key")] = None):
        store, session, state = prepare_chat(body, request, key)
        stop = Event()

        def event(type, **fields):
            return "data: " + json.dumps({"type": type, **fields}, ensure_ascii=False) + "\n\n"

        async def events():
            try:
                yield event("status", phase="routing")
                stream = request.app.state.workflow.astream({**state, "stream_stop": stop}, stream_mode=["custom", "values"])
                async with aclosing(stream):
                    try:
                        async for mode, chunk in stream:
                            if mode == "custom":
                                yield event(**chunk)
                            else:
                                final = chunk
                    finally:
                        stop.set()
                result = await run_in_threadpool(save_chat, store, session, body, final)
                yield event("done", result=result)
            except HTTPException as exc:
                yield event("error", status=exc.status_code, detail=exc.detail)
            except Exception:
                log.exception("Chat stream failed")
                yield event("error", status=500, detail="问答未完成，请重试。")
            finally:
                stop.set()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

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

    @app.get("/api/memory/assessments")
    def assessment_memory(request: Request, melody_id: Melody | None = None, segment_id: Identifier | None = None,
                          query: str | None = Query(None, min_length=1, max_length=300),
                          limit: int = Query(10, ge=1, le=50)):
        memory = request.app.state.memory
        if segment_id is not None:
            history = memory.segment_history(segment_id, limit)
            if history is None:
                raise HTTPException(404, "该片段暂无有效测评记忆。")
            return history
        if melody_id is None:
            melody_id = memory.resolve_melody(query or "")
        if melody_id is None:
            raise HTTPException(422, "无法确定参考旋律，请指定 melody_id 或填写明确的旋律名称。")
        return memory.assessment_history(melody_id, limit)

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
