"""Black-box two-turn evaluation with synthetic inputs and real model decisions.

Run from the repository root with python -m eval.agent_workflows.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.parse import urlparse
import wave
from io import BytesIO

from fastapi.testclient import TestClient

from deepvocal import audio
from deepvocal.assessment import report_metrics
from deepvocal.assessment_agent import ALIGN, ALIGN_DTW, ANALYZE, FINISH, REUSE, SEARCH
from deepvocal.audio_alignment import DEFAULT_AUDIO_MODEL
from deepvocal.main import choose_model, create_app
from deepvocal.practice_agent import BUILD as BUILD_PLAN, FINISH as FINISH_PLAN, SEARCH as SEARCH_REPORT


PHRASE = [60, 63, 67, 65, 69, 64, 61]
REFERENCE = [49, 51, 54] + PHRASE + [74, 72, 70]
TITLE = "晨雾练习曲"
COMPLETE = "completed"
INPUT = "needs_input"
CASES = (
    ("Q1", "qa", "换声区定义 → 省略主题的训练追问"),
    ("Q2", "qa", "唇颤音热身 → 吹不动时的替代动作"),
    ("Q3", "qa", "寒暄 → 致谢，两轮均无需检索"),
    ("A1", "assessment", "内置旋律测评 → 复用已有报告"),
    ("A2", "assessment", "缺歌名 → 补歌名并复用原录音"),
    ("A3", "assessment", "曲库未命中 → 上传参考并测评"),
    ("A4", "assessment", "备用 DTW 选择候选 → 按片段测评"),
    ("A5", "assessment", "附件用音频模型对齐 → 复用测评报告"),
    ("P1", "practice_plan", "换声练习 → 自然语言要求每天 8 分钟"),
    ("P2", "practice_plan", "音高偏低计划 → 目标改为节奏"),
    ("P3", "practice_plan", "通用基准 → 只练长音稳定性"),
    ("P4", "practice_plan", "未找到指定歌曲测评 → 补齐后用定性报告规划"),
    ("S1", "singing", "结构化速度与移调更改 → 比对实际 WAV"),
    ("S2", "singing", "自然语言续用旋律并减速 → 比对实际 WAV"),
    ("S3", "singing", "无合成配置时请求歌词 → 改用示范旋律"),
    ("C1", "combined", "问答及测评后制定计划 → 按原报告重规划"),
)


def check(checks):
    return {"pass": all(checks.values()), "checks": checks}


class Session:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.session_id = None
        self.turns = []

    def upload(self, notes, *, reference=False, title=TITLE, transpose=0):
        wav = audio.guide(notes, 90, transpose)
        endpoint = "/api/audio/reference" if reference else "/api/audio"
        response = self.client.post(endpoint, data={"song_title": title} if reference else {},
                                    files={"file": ("synthetic.wav", wav, "audio/wav")})
        response.raise_for_status()
        return response.json()["id"]

    def ask(self, query, intent, **fields):
        started = monotonic()
        response = self.client.post("/api/chat", json={"query": query, "intent": intent,
            "session_id": self.session_id, "provider": self.model.provider, "model": self.model.name, **fields},
            headers={"X-Model-API-Key": self.model.key})
        response.raise_for_status()
        result = response.json()
        self.session_id = result["session_id"]
        job = {}
        if pending := result.get("job"):
            while monotonic() - started < 220:
                status = self.client.get("/api/jobs/" + pending["id"])
                status.raise_for_status()
                job = status.json()
                if job["state"] in ("succeeded", "failed"):
                    break
                sleep(0.3)
        turn = {"query": query, "requested_intent": intent, "steps": result.get("steps", []),
                "task_statuses": (job.get("result") or {}).get("task_statuses", result.get("task_statuses", [])),
                "job_state": job.get("state"), "latency_seconds": round(monotonic() - started, 2)}
        self.turns.append(turn)
        return result, job, turn

    def knowledge(self, query, *, topic=None, evidence=(), answer=None):
        result, _, turn = self.ask(query, "qa")
        score_qa(turn, result, topic, evidence, answer)

    def assess(self, query, expected, **fields):
        result, job, turn = self.ask(query, "assessment", **fields)
        report = job.get("result") or {}
        score_assessment(turn, report, expected)
        return job, report, turn

    def plan(self, query, focus, **fields):
        result, _, turn = self.ask(query, "practice_plan", **fields)
        score_plan(turn, result.get("plan"), focus)
        score_planning_tools(turn, result.get("planning_agent", {}),
                             linked=bool(fields.get("assessment_job_id")))
        return result.get("plan") or {}, turn

    def guide(self, query, melody, bpm, transpose, *, fields=None, unavailable=False):
        result, job, turn = self.ask(query, "auto" if fields is None else "singing", **(fields or {}))
        artifact = job.get("result") or {}
        if unavailable:
            turn["singing"] = check({"routed": result["steps"] == ["singing"], "missing_synthesis_config":
                job.get("state") == "failed" and "DEEPVOCAL_SINGING_URL" in (job.get("error") or "")})
            turn["singing"]["expected_outcome"] = "needs_synthesis_configuration"
            return
        expected = audio.guide(audio.MELODIES[melody]["notes"], bpm, transpose)
        rendered = b""
        if artifact.get("audio_url"):
            response = self.client.get(artifact["audio_url"])
            if response.is_success:
                rendered = response.content
        turn["singing"] = check({"routed": result["steps"] == ["singing"], "playable_guide":
                                 job.get("state") == "succeeded" and artifact.get("type") == "guide" and bool(rendered),
                                 "requested_melody_tempo_key": rendered == expected})
        for name, wav in (("actual_seconds", rendered), ("expected_seconds", expected)):
            if wav:
                with wave.open(BytesIO(wav)) as stream:
                    turn["singing"][name] = round(stream.getnframes() / stream.getframerate(), 3)


def score_qa(turn, result, topic, evidence=(), answer=None):
    rag = result.get("rag", {})
    mode = result.get("qa_generation", {}).get("mode")
    ids = [item["id"] for item in result.get("evidence", [])]
    decision = bool(rag.get("used")) == bool(topic) and mode in ("model", "model_fallback")
    grounded_query = not topic or bool(re.search(topic, rag.get("query", "")))
    turn["qa"] = check({"correct_tool_decision": decision, "query_targets_topic": grounded_query,
        "relevant_citation": not evidence or bool(set(evidence) & set(ids)),
        "answer_matches_expected": bool(result.get("answer")) and (not answer or bool(re.search(answer, result["answer"])))})
    turn["qa"].update(expected_search=bool(topic), search=bool(rag.get("used")), query=rag.get("query"),
                      generation=mode, citation_ids=ids, answer=result.get("answer"),
                      correct_search=bool(topic and rag.get("used") and grounded_query and rag.get("tool") == "search_knowledge"))


def score_assessment(turn, report, expected):
    trace = report.get("assessment_agent", {})
    actions = [tuple(item[key] for key in ("tool", "status")) for item in trace.get("actions", [])]
    turn["assessment"] = check({"terminal_status": trace.get("status") == expected[-1][1]})
    turn["assessment"].update(expected=expected, actions=actions, decisions=trace.get("turns", 0),
                              terminal=trace.get("status"), issue=report.get("issue"))


def require(turn, agent, **checks):
    current = turn[agent]
    current["checks"].update(checks)
    current["pass"] = all(current["checks"].values())


def score_plan(turn, value, focus, source=None):
    plan = value or {}
    days, minutes = plan.get("days"), plan.get("minutes_per_day")
    sessions = plan.get("sessions", [])
    generation = plan.get("generation", {})
    criteria = plan.get("retest", {}).get("criteria", [])
    turn["practice_plan"] = check({
        "focus": plan.get("focus") == focus,
        "schedule": bool(days and minutes and len(sessions) == days
                         and all(sum(item["minutes"] for item in day["exercises"]) == minutes for day in sessions)),
        "retest": any(item["evidence_id"] == focus for item in criteria),
        "source": not source or plan.get("source_assessment") == source,
    })
    turn["practice_plan"].update(generation=generation, focus=plan.get("focus"), days=days, minutes_per_day=minutes)


def score_planning_tools(turn, trace, *, linked=False, missing=False):
    expected = [(SEARCH_REPORT, INPUT), (FINISH_PLAN, INPUT)] if missing else [
        *([(SEARCH_REPORT, "found")] if linked else []), (BUILD_PLAN, "validated"), (FINISH_PLAN, COMPLETE)]
    actions = [(item["tool"], item["status"]) for item in trace.get("actions", [])]
    correct = 0
    for action in actions:
        if correct < len(expected) and action == expected[correct]:
            correct += 1
    turn["planning_tools"] = {"expected": expected, "actions": actions, "correct": correct,
                              "decisions": trace.get("turns", 0), "finalized_by": trace.get("finalized_by")}


def scenario(key, session):
    s = session
    if key == "Q1":
        s.knowledge("什么是换声区？请给一条出处。", topic=r"换声|声区", evidence=("register", "falsetto-practice"), answer="声区")
        s.knowledge("初学者怎么练它？给我一个具体动作。", topic=r"换声|声区", evidence=("register", "falsetto-practice"), answer="滑音|相邻音")
    elif key == "Q2":
        s.knowledge("怎么做唇颤音热身？", topic=r"唇.{0,2}[颤震]|唇颤", evidence=("lip", "laryngeal-coordination-practice"), answer="唇")
        s.knowledge("如果嘴唇吹不动，可以换成什么发声？", topic=r"唇.{0,2}[颤震]|哼鸣", evidence=("lip", "laryngeal-coordination-practice"), answer="哼鸣")
    elif key == "Q3":
        s.knowledge("你好，很高兴见到你。")
        s.knowledge("谢谢你，今天先聊到这里。")
    elif key == "A1":
        take = s.upload(audio.MELODIES["warmup"]["notes"], transpose=-1)
        job, report, turn = s.assess("请用内置八音热身测这段录音。", [(ANALYZE, COMPLETE), (FINISH, COMPLETE)],
                                     audio_id=take, reference_mode="builtin", alignment_mode="signal")
        require(turn, "assessment", pitch_flat=report.get("issue") == "pitch_flat",
                pitch_measured=(report.get("median_pitch_error_cents") or 0) < -80)
        _, reused, turn = s.assess("请复用刚才的测评结果。", [(REUSE, COMPLETE), (FINISH, COMPLETE)],
                                   assessment_job_id=job.get("id"))
        require(turn, "assessment", original_report=reused.get("source_assessment") == job.get("id"))
    elif key in ("A2", "A3"):
        take = s.upload(PHRASE)
        reference = s.upload(REFERENCE, reference=True) if key == "A2" else None
        prompt = "帮我测评这段录音。" if key == "A2" else "测评《未入库练习曲》。"
        _, report, turn = s.assess(prompt, [(SEARCH, INPUT), (FINISH, INPUT)], audio_id=take, alignment_mode="signal")
        require(turn, "assessment", correct_next_input=report.get("issue") ==
                ("song_title_required" if key == "A2" else "reference_not_found"))
        if key == "A2":
            _, report, turn = s.assess(f"我唱的是《{TITLE}》。", [(SEARCH, "reference_found"), (ANALYZE, COMPLETE),
                (FINISH, COMPLETE)], audio_id=take, alignment_mode="signal")
        else:
            reference = s.upload(REFERENCE, reference=True)
            _, report, turn = s.assess("我上传了对比片段，请再测一次。", [(ANALYZE, COMPLETE), (FINISH, COMPLETE)],
                                        audio_id=take, reference_audio_id=reference, alignment_mode="signal")
        require(turn, "assessment", same_recording=report.get("recording_audio_id") == take,
                right_reference=(report.get("reference") or {}).get("audio_id") == reference)
    elif key == "A4":
        take = s.upload(PHRASE)
        reference = s.upload(REFERENCE, reference=True)
        _, report, turn = s.assess("用备用对齐给我列出对应片段。", [(ALIGN_DTW, INPUT), (FINISH, INPUT)],
                                   audio_id=take, reference_audio_id=reference, alignment_mode="dtw")
        candidates = report.get("candidates") or []
        require(turn, "assessment", has_candidate=bool(candidates))
        first = candidates[0] if candidates else {"start_seconds": 2, "end_seconds": 6.67}
        _, result, turn = s.assess("我选这个片段，现在测评。", [(ANALYZE, COMPLETE), (FINISH, COMPLETE)],
                                   audio_id=take, reference_audio_id=reference, alignment_mode="signal",
                                   reference_start_seconds=first["start_seconds"], reference_end_seconds=first["end_seconds"])
        require(turn, "assessment", right_reference=(result.get("reference") or {}).get("audio_id") == reference)
    elif key == "A5":
        take = s.upload(PHRASE)
        reference = s.upload(REFERENCE, reference=True)
        job, report, turn = s.assess("请先对齐我上传的参考音频，再测这段演唱。",
                                     [(ALIGN, "aligned"), (ANALYZE, COMPLETE), (FINISH, COMPLETE)],
                                     audio_id=take, reference_audio_id=reference, alignment_mode="llm")
        require(turn, "assessment", model_audio_feedback=bool(report.get("audio_feedback")),
                right_reference=(report.get("reference") or {}).get("audio_id") == reference)
        _, result, turn = s.assess("再给我看刚才那份测评。", [(REUSE, COMPLETE), (FINISH, COMPLETE)],
                                   assessment_job_id=job.get("id"))
        require(turn, "assessment", original_report=result.get("source_assessment") == job.get("id"))
    elif key == "P1":
        s.plan("我想练稳换声区，给一份计划。", "register")
        plan, turn = s.plan("换声区继续练，但每天只有 8 分钟。", "register")
        require(turn, "practice_plan", eight_minutes=plan.get("minutes_per_day") == 8)
    elif key == "P2":
        s.plan("请安排改善音高偏低的练习计划。", "pitch_flat")
        s.plan("把练习目标改成改善拖拍。", "timing")
    elif key == "P3":
        s.plan("我刚开始练歌，给我一个入门练习安排。", "general")
        s.plan("下一步只练长音稳定性。", "stability")
    elif key == "P4":
        result, _, turn = s.ask("根据刚才《山谷微光》的测评制定练习计划。", "practice_plan")
        turn["practice_plan"] = check({"asks_for_assessment": result["task_statuses"][0]["status"] == INPUT,
                                       "no_ungrounded_plan": "plan" not in result})
        score_planning_tools(turn, result.get("planning_agent", {}), missing=True)
        store = s.client.app.state.store
        with store.connect() as db:
            report_id = store.enqueue("assessment", {}, db)
            store.finish({"id": report_id, "kind": "assessment"}, report_metrics("stability") | {
                "assessment_method": "audio_model",
                "reference": {"title": "山谷微光-合成练习.wav", "segment_start_seconds": 18.5, "segment_end_seconds": 24.0},
                "audio_feedback": {"stability": "句尾持续音有下坠。"},
            }, db=db)
        result, _, turn = s.ask("现在按上次《山谷微光》的测评规划，只练长音稳定性，每天最多8分钟。", "practice_plan",
                                 assessment_job_id="stale-session-report", assessment_source="session")
        plan = result.get("plan") or {}
        score_plan(turn, plan, "stability", report_id)
        score_planning_tools(turn, result.get("planning_agent", {}), linked=True)
        require(turn, "practice_plan", duration_cap=0 < (plan.get("minutes_per_day") or 0) <= 8,
                original_segment=plan.get("retest", {}).get("reference", {}).get("start_seconds") == 18.5)
    elif key == "S1":
        s.guide("生成五音琶音示范。", "triad", 95, 0, fields={"melody_id": "triad", "bpm": 95})
        s.guide("同一旋律降低两个半音并放慢。", "triad", 75, -2,
                fields={"melody_id": "triad", "bpm": 75, "transpose": -2})
    elif key == "S2":
        s.guide("生成五音琶音示范。", "triad", 90, 0, fields={"melody_id": "triad"})
        s.guide("把刚才那段旋律放慢到 70 BPM，再生成示范音。", "triad", 70, 0)
    elif key == "S3":
        s.guide("用歌词给我一段示范唱。", "warmup", 90, 0,
                fields={"lyrics": "窗外起了风，我轻声唱"}, unavailable=True)
        s.guide("那就给我八音热身的参考旋律。", "warmup", 90, 0, fields={})
    elif key == "C1":
        take = s.upload(audio.MELODIES["warmup"]["notes"], transpose=-1)
        result, job, turn = s.ask("什么是换声区？请测评这段录音，并根据测评制定练习计划。",
                                 "auto", audio_id=take, reference_mode="builtin", alignment_mode="signal")
        report = job.get("result") or {}
        score_qa(turn, result, r"声区|换声", ("register", "falsetto-practice"), "声区")
        score_assessment(turn, report, [(ANALYZE, COMPLETE), (FINISH, COMPLETE)])
        require(turn, "assessment", pitch_flat=report.get("issue") == "pitch_flat")
        score_plan(turn, report.get("plan"), "pitch_flat", job.get("id"))
        score_planning_tools(turn, report.get("planning_agent", {}), linked=True)
        turn["routing"] = check({"all_requested_agents": set(turn["steps"]) == {"qa", "assessment", "practice_plan"}})
        flags = {"assessment_job_id": job["id"]} if job.get("id") else {}
        plan, turn = s.plan("根据刚才的测评，这次只练长音稳定性。", "stability", **flags)
        require(turn, "practice_plan", original_report=bool(job.get("id")) and plan.get("source_assessment") == job["id"])


def run_case(case, model):
    key, agent, name = case
    started = monotonic()
    turns, error = [], None
    with TemporaryDirectory(prefix="deepvocal-agents-") as tmp, TestClient(create_app(tmp)) as client:
        session = Session(client, model)
        try:
            scenario(key, session)
        except Exception as exc:
            error = type(exc).__name__
            if hasattr(exc, "response"):
                error += f" HTTP {exc.response.status_code}"
        turns = session.turns
    passed = len(turns) == 2 and not error and all(
        all(turn.get(agent, {}).get("pass", False) for agent in turn.keys() &
            {"qa", "assessment", "practice_plan", "singing", "routing"})
        and any(agent in turn for agent in ("qa", "assessment", "practice_plan", "singing")) for turn in turns)
    return {"id": key, "agent": agent, "scenario": name, "pass": bool(passed), "turns": turns,
            "error": error, "seconds": round(monotonic() - started, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    root = Path.cwd()
    for key in [key for key in os.environ if key.startswith("DEEPVOCAL_")]:
        del os.environ[key]
    os.environ["DEEPVOCAL_RMVPE_ONNX_PATH"] = str(root / "data/models/rmvpe.onnx")
    with sqlite3.connect("file:data/deepvocal.sqlite3?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        model = choose_model("dashscope", None, None, db.execute(
            "SELECT provider, model, protected_key FROM model_settings WHERE id=1").fetchone())
    if not model or urlparse(model.endpoint()).hostname != "dashscope.aliyuncs.com":
        raise SystemExit("Configure DashScope before running the live evaluation.")
    selected = [case for case in CASES if not args.cases or case[0] in args.cases]
    output = {"started": datetime.now().astimezone().isoformat(timespec="seconds"),
              "text_model": model.name, "audio_model": DEFAULT_AUDIO_MODEL, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(lambda case: run_case(case, model), selected):
            output["cases"].append(row)
            print(f"{row['id']} completed={row['pass']} turns={len(row['turns'])} seconds={row['seconds']}", flush=True)
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scores = [turn["planning_tools"] for case in output["cases"] for turn in case["turns"] if "planning_tools" in turn]
    if scores:
        output["planning_tool_totals"] = {
            "correct": sum(row["correct"] for row in scores),
            "decisions": sum(row["decisions"] for row in scores),
            "expected_steps": sum(len(row["expected"]) for row in scores),
        }
        print("Planning tools", output["planning_tool_totals"], flush=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
