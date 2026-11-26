from contextlib import nullcontext
import io
import json
import os
from base64 import b64decode
from fractions import Fraction
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import time
import unittest
from unittest.mock import patch

import av
from fastapi.testclient import TestClient
import httpx
import numpy as np

from deepvocal import audio
from deepvocal.assessment import measure_song, report_metrics
from deepvocal.assessment_agent import (
    ALIGN, ALIGN_DTW, ANALYZE, FINISH, MAX_TURNS, REUSE, SEARCH, TOOLS, inspect_call, valid_args,
)
from deepvocal.audio_alignment import cut_wave, refine_with_lyrics
from deepvocal.dtw_alignment import subsequence_candidates
from deepvocal.main import assessment_model_for, create_app
from deepvocal.memory import MemoryService
from deepvocal.model import Model, ModelError
from deepvocal.practice_agent import BUILD as BUILD_PLAN, FINISH as FINISH_PLAN, SEARCH as SEARCH_REPORT
from deepvocal.practice_agent import BuildArgs, SearchArgs, TOOLS as PLAN_TOOLS
from deepvocal.workflow import SEARCH_TOOLS, resume_assessment, search_request


def tool_reply(query, name="search_knowledge"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_search", "type": "function",
        "function": {"name": name, "arguments": json.dumps({"query": query})},
    }]}


def assessment_tool(name, args=None, call_id="assessment_call"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }]}


def planning_reply(messages, proposal):
    initial = json.loads(messages[1]["content"])
    if messages[-1]["role"] != "tool":
        if initial["assessment_required"]:
            return assessment_tool(SEARCH_REPORT, {"scope": initial["suggested_scope"]}, "search_report")
        context, reference = initial["planning"], ""
    else:
        result = json.loads(messages[-1]["content"])
        if result["status"] in ("validated", "needs_input"):
            status = "completed" if result["status"] == "validated" else "needs_input"
            return assessment_tool(FINISH_PLAN, {"status": status}, "submit_plan")
        report = next((item for item in reversed(messages) if item["role"] == "tool"
                       and json.loads(item["content"])["status"] == "found"), None)
        result = json.loads(report["content"]) if report else {}
        context, reference = result.get("planning", initial["planning"]), result.get("assessment_ref", "")
    draft = proposal(context) if callable(proposal) else proposal
    return assessment_tool(BUILD_PLAN, {"focus": context["primary_evidence_id"], "assessment_ref": reference,
                                        "draft": draft}, f"build_plan_{len(messages)}")


def encode_m4a(samples, rate, codec="aac"):
    output = io.BytesIO()
    with av.open(output, "w", format="ipod") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples[None, :].astype("float32"), format="fltp", layout="mono")
        frame.sample_rate = rate
        frame.pts = 0
        frame.time_base = Fraction(1, rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def assessment_model_output(model, messages, *, tools=None):
    latest = messages[-1]
    if latest["role"] == "tool":
        observed = json.loads(latest["content"])
        if observed["status"] == "reference_found":
            if "references" in observed:
                mode = json.loads(messages[1]["content"])["alignment_mode"]
                return assessment_tool(ALIGN_DTW if mode == "dtw" else ALIGN, call_id="align")
            return assessment_tool(ANALYZE, call_id="analyze")
        if observed["status"] == "aligned":
            return assessment_tool(ANALYZE, call_id="analyze")
        return assessment_tool(FINISH, {"status": observed["status"]}, "finish")
    task = json.loads(latest["content"])
    if task["has_requested_report"]:
        return assessment_tool(REUSE)
    if task["reference_kind"] == "unspecified":
        return assessment_tool(SEARCH)
    if task["reference_kind"] == "uploaded_song" and not task["has_selected_range"]:
        if task["alignment_mode"] == "llm":
            return assessment_tool(ALIGN)
        if task["alignment_mode"] == "dtw":
            return assessment_tool(ALIGN_DTW)
    return assessment_tool(ANALYZE)


class ToolRegistration(unittest.TestCase):
    def test_each_tool_documents_when_to_call_and_has_a_valid_example(self):
        for spec in [*SEARCH_TOOLS, *TOOLS, *PLAN_TOOLS]:
            function = spec["function"]
            name = function["name"]
            with self.subTest(tool=name):
                description = function["description"]
                self.assertIn("应调用的情况", description)
                self.assertIn("不应调用的情况", description)
                self.assertEqual(description.count("示例中"), 1)
                _, marker, example = description.rpartition(f"发起 {name}(")
                self.assertTrue(marker)
                arguments = json.loads(example.removesuffix(") 的结构化工具调用。"))
                parameters = function["parameters"]
                self.assertTrue(set(parameters["required"]) <= arguments.keys() <= parameters["properties"].keys())
                if name == "search_knowledge":
                    self.assertEqual(search_request(tool_reply(arguments["query"]))[1], arguments["query"])
                elif name in (SEARCH_REPORT, BUILD_PLAN, FINISH_PLAN):
                    self.assertEqual(inspect_call(assessment_tool(name, arguments))["args"], arguments)
                    if name == SEARCH_REPORT:
                        SearchArgs.model_validate(arguments)
                    elif name == BUILD_PLAN:
                        BuildArgs.model_validate(arguments)
                    else:
                        self.assertIn(arguments["status"], parameters["properties"]["status"]["enum"])
                else:
                    self.assertEqual(inspect_call(assessment_tool(name, arguments))["args"], arguments)
                    if name == FINISH:
                        self.assertIn(arguments["status"], parameters["properties"]["status"]["enum"])
                    else:
                        self.assertTrue(valid_args(name, arguments))


class DeepVocalAPI(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch("deepvocal.main.assessment_model_for", side_effect=
                                lambda store, body, key, client: (client or Model("custom", "fixture", ""), "request")))
        self.enterContext(patch("deepvocal.assessment_agent.model_step", side_effect=assessment_model_output))
        directory = self.enterContext(TemporaryDirectory())
        self.client = self.enterContext(TestClient(create_app(directory)))

    def wait_for(self, condition):
        for _ in range(100):
            result = condition()
            if result:
                return result
            time.sleep(0.05)
        self.fail("Worker did not finish")

    def status(self, job_id):
        job = self.client.get(f"/api/jobs/{job_id}").json()
        return job if job["state"] in ("succeeded", "failed") else None

    def qa(self, query, **fields):
        def respond(messages, *, tool_choice, **kwargs):
            if tool_choice == "auto":
                return tool_reply(query)
            evidence = json.loads(messages[-1]["content"])["evidence"]
            return {"content": f"示例回答 [{evidence[0]['id']}]"}

        with patch("deepvocal.model.Model.chat_message", side_effect=respond):
            return self.client.post("/api/chat", json={
                "query": query, "intent": "qa", "provider": "dashscope", "model": "qwen-plus", **fields,
            }, headers={"X-Model-API-Key": "fixture-key"}).json()

    def test_recording_plan_and_cross_session_memory(self):
        melody = self.client.get("/api/melodies/warmup/guide.wav?transpose=-1")
        recording = self.client.post("/api/audio", files={"file": ("guide.wav", melody.content, "audio/wav")})
        self.assertEqual(recording.status_code, 201)
        chat = self.client.post("/api/chat", json={
            "query": "测评并安排练习", "audio_id": recording.json()["id"], "reference_mode": "builtin",
            "days": 2, "minutes": 10,
        }).json()
        self.assertEqual(chat["steps"], ["assessment", "practice_plan"])
        job = self.wait_for(lambda: self.status(chat["job"]["id"]))
        self.assertEqual(job["state"], "succeeded", job)
        report = job["result"]
        self.assertEqual(report["matched_notes"], 8)
        self.assertAlmostEqual(report["median_pitch_error_cents"], -100, delta=10)
        self.assertEqual(report["plan"]["focus"], "pitch_flat")
        self.assertEqual(report["plan"]["source_assessment"], job["id"])
        self.assertEqual(report["plan"]["retest"]["reference"]["bpm"], 90)
        pitch_criterion = report["plan"]["retest"]["criteria"][0]
        self.assertEqual(pitch_criterion["comparator"], "closer_to_zero")
        self.assertEqual(pitch_criterion["baseline"]["value"], report["median_pitch_error_cents"])
        self.assertEqual(len(report["plan"]["sessions"]), 2)
        for day in report["plan"]["sessions"]:
            self.assertEqual(sum(e["minutes"] for e in day["exercises"]), 10)
        self.assertEqual(self.client.post("/api/memory/progress", json={
            "melody_id": "warmup", "completed_days": 2,
        }).status_code, 202)
        self.wait_for(lambda: len(self.client.get("/api/memory").json()["facts"]) >= 3)
        for _ in range(2):
            new = self.client.post("/api/chat", json={"query": "为我安排练习", "intent": "practice_plan"}).json()
            self.assertNotEqual(new["session_id"], chat["session_id"])
            self.assertEqual(new["plan"]["focus"], "pitch_flat")
            self.assertEqual(new["plan"]["song_progress"]["completed_days"], 2)
            self.assertEqual(new["plan"]["minutes_per_day"], 10)
        draft = self.client.post("/api/skills/candidates", json={"focus": "pitch_flat"}).json()
        url = f"/api/skills/pitch_flat/{draft['version']}"
        self.assertEqual(len(self.client.get(url).json()["trace_ids"]), 2)
        self.assertEqual(self.client.post(url + "/review", json={"approve": True}).json()["state"], "active")
        latest = self.client.post("/api/chat", json={"query": "安排练习", "intent": "practice_plan"}).json()
        self.assertEqual(latest["plan"]["skill_version"], 2)

    def test_assessment_memory_keeps_runs_and_uses_only_comparable_reports(self):
        def assess(wav, bpm=90, *, also_plan=False):
            uploaded = self.client.post("/api/audio", files={"file": ("take.wav", wav, "audio/wav")}).json()
            result = self.client.post("/api/chat", json={
                "query": "测评并安排练习" if also_plan else "测评",
                "intent": "auto" if also_plan else "assessment", "audio_id": uploaded["id"],
                "reference_mode": "builtin", "bpm": bpm,
            }).json()
            job = self.wait_for(lambda: self.status(result["job"]["id"]))
            self.assertEqual(job["state"], "succeeded", job)
            return job

        flat = self.client.get("/api/melodies/warmup/guide.wav?transpose=-1").content
        faster = self.client.get("/api/melodies/warmup/guide.wav?bpm=105").content
        in_tune = self.client.get("/api/melodies/warmup/guide.wav").content
        first = assess(flat)
        other_tempo = assess(faster, bpm=105)
        partial = assess(audio.guide(audio.MELODIES["warmup"]["notes"][:5], 90, 0))
        latest = assess(in_tune)
        comparison = latest["result"]["comparison"]
        self.assertEqual(comparison["previous"]["report_id"], first["id"])
        self.assertNotIn("comparison", other_tempo["result"])
        self.assertEqual(partial["result"]["matched_notes"], 5)
        self.assertNotIn("comparison", partial["result"])
        self.assertAlmostEqual(comparison["absolute_bias_reduction_cents"], 100, delta=10)

        silent = audio.to_wave(np.zeros(8000), 8000)
        insufficient = assess(silent, also_plan=True)["result"]
        self.assertEqual(insufficient["issue"], "insufficient_audio")
        self.assertIn("plan_error", insufficient)
        self.assertNotIn("plan", insufficient)

        response = self.client.get("/api/memory/assessments", params={"query": "八音热身", "limit": 2})
        self.assertEqual(response.status_code, 200, response.text)
        history = response.json()
        self.assertEqual(history["topic"]["id"], "assessment:warmup")
        self.assertEqual([row["report_id"] for row in history["observations"]], [latest["id"], partial["id"]])
        self.assertEqual(history["comparison"], comparison)
        self.assertNotIn("notes", history["observations"][0])
        self.assertEqual(self.client.get("/api/memory/assessments?melody_id=triad").json()["observations"], [])
        self.assertEqual(self.client.get("/api/memory/assessments", params={"query": "八音练习和五音练习"}).status_code, 422)
        plan = self.client.post("/api/chat", json={"query": "安排练习", "intent": "practice_plan"}).json()["plan"]
        self.assertEqual(plan["source_assessment"], latest["id"])

    def test_assessment_memory_backfills_older_reports_idempotently(self):
        store = self.client.app.state.store
        spec = {"audio_id": "older-audio", "melody_id": "triad", "bpm": 80, "transpose": 2, "offset_seconds": 0.25}
        wav = audio.guide(audio.MELODIES["triad"]["notes"], 80, 2)
        report = audio.assess(audio.to_wave(np.pad(audio.read_wave(wav)[0], (5513, 0)), 22050), spec)
        del report["scoring_version"]
        with store.connect() as db:
            job_id = store.enqueue("assessment", spec, db)
            store.finish({"id": job_id, "kind": "assessment"}, report, db=db)
        memory = self.client.app.state.memory
        memory.backfill_assessments()
        memory.backfill_assessments()
        result = self.client.get("/api/memory/assessments?melody_id=triad").json()
        self.assertEqual(len(result["observations"]), 1)
        row = result["observations"][0]
        self.assertEqual((row["report_id"], row["bpm"], row["transpose"], row["offset_seconds"], row["scoring_version"]),
                         (job_id, 80, 2, 0.25, 1))

    def test_memory_query_uses_one_semantic_pass_when_aliases_miss(self):
        def embed(texts):
            if len(texts) == 2:
                return [[1, 0], [0, 1]]
            return [[1, 1]] if texts[0] == "含糊的查询" else [[0.05, 1]]

        with patch.object(self.client.app.state, "memory",
                          MemoryService(self.client.app.state.store, embed)):
            match = self.client.get("/api/memory/assessments", params={"query": "五个音的琶音"})
            ambiguous = self.client.get("/api/memory/assessments", params={"query": "含糊的查询"})
        self.assertEqual(match.json()["topic"]["melody_id"], "triad")
        self.assertEqual(ambiguous.status_code, 422)

    def song_recording(self, notes, *, transpose=0, offset=0, bpm=90):
        wav = audio.guide(notes, bpm, transpose)
        signal, rate = audio.read_wave(wav, audio.MAX_REFERENCE_SECONDS)
        return audio.to_wave(np.pad(signal, (round(offset * rate), 0)), rate)

    def upload_song_audio(self, wav, kind="", *, filename="song.wav", song_title=None):
        form = {"song_title": song_title} if song_title else {}
        response = self.client.post("/api/audio" + kind, data=form, files={"file": (filename, wav, "audio/wav")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def assess_song(self, reference, fragment, **settings):
        query = settings.pop("query", "测评")
        response = self.client.post("/api/chat", json={
            "intent": "auto" if "计划" in query else "assessment", "query": query,
            "audio_id": fragment["id"], "reference_audio_id": reference["id"], "alignment_mode": "signal", **settings,
        })
        self.assertEqual(response.status_code, 200, response.text)
        job = self.wait_for(lambda: self.status(response.json()["job"]["id"]))
        self.assertEqual(job["state"], "succeeded", job)
        return job

    def assess_from_library(self, recording, query="请测评这段录音", **fields):
        response = self.client.post("/api/chat", json={"query": query, "audio_id": recording["id"],
                                                       "alignment_mode": "signal", **fields})
        self.assertEqual(response.status_code, 200, response.text)
        job = self.wait_for(lambda: self.status(response.json()["job"]["id"]))
        self.assertEqual(job["state"], "succeeded", job)
        return job["result"]

    def test_m4a_uploads_are_playable_and_can_be_assessed_from_the_library(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        original = self.song_recording([48, 52] + phrase + [74])
        source, source_rate = audio.read_wave(original)
        sung, sung_rate = audio.read_wave(self.song_recording(phrase))
        reference = self.client.post("/api/audio/reference", files={
            "file": ("晴天-歌手.m4a", encode_m4a(source, source_rate, "alac"), "audio/mp4"),
        })
        recording = self.client.post("/api/audio", files={
            "file": ("录音.M4A", encode_m4a(sung, sung_rate), "audio/x-m4a"),
        })
        self.assertEqual(reference.status_code, 201, reference.text)
        self.assertEqual(recording.status_code, 201, recording.text)
        self.assertAlmostEqual(reference.json()["duration_seconds"], len(source) / source_rate)
        for entry in (reference.json(), recording.json()):
            playable = self.client.get(entry["audio_url"])
            self.assertEqual(playable.headers["content-type"], "audio/wav")
            self.assertTrue(playable.content.startswith(b"RIFF"))
        report = self.assess_from_library(recording.json(), song_title="晴天")
        self.assertEqual(report["reference"]["audio_id"], reference.json()["id"])
        self.assertGreaterEqual(report["matched_notes"], 6)

    def test_m4a_upload_rejects_invalid_files_and_checks_decoded_duration(self):
        rate = 8000
        for suffix in ("", "/reference"):
            invalid = self.client.post("/api/audio" + suffix, files={
                "file": ("损坏.m4a", b"not audio", "audio/mp4"),
            })
            self.assertEqual(invalid.status_code, 422, invalid.text)
        for duration, status in ((60, 201), (61, 422)):
            response = self.client.post("/api/audio", files={
                "file": ("录音.m4a", encode_m4a(np.zeros(duration * rate), rate), "audio/mp4"),
            })
            self.assertEqual(response.status_code, status, response.text)

    def test_library_finds_earlier_upload_and_scores_against_original_key(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        intro = [52, 56, 54, 57, 52] * 20
        original = self.song_recording(intro + phrase + [72, 74, 72])
        store = self.client.app.state.store
        store.put_audio("legacy-song", "Legacy song.wav", "reference", original)
        self.upload_song_audio(original, "/reference")
        self.upload_song_audio(self.song_recording([72, 68, 74, 76, 71, 75, 70]), "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase, transpose=2, bpm=99))
        report = self.assess_from_library(recording, song_title="Legacy song")
        self.assertEqual(report["reference"]["selection"], "library")
        self.assertEqual(report["reference"]["audio_id"], "legacy-song")
        self.assertEqual(report["reference"]["detected_transpose"], 2)
        self.assertAlmostEqual(report["reference"]["start_seconds"], len(intro) * (60 / 90), delta=0.16)
        self.assertGreaterEqual(report["matched_notes"], 6)
        self.assertAlmostEqual(report["median_pitch_error_cents"], 200, delta=15)
        self.assertEqual(report["task_statuses"][0]["status"], "completed")
        self.assertEqual(report["assessment_agent"]["actions"], [
            {"tool": SEARCH, "status": "reference_found"},
            {"tool": ANALYZE, "status": "completed"},
            {"tool": FINISH, "status": "completed"},
        ])

    def test_library_miss_stops_and_allows_resubmitting_same_recording_with_reference(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        recording = self.upload_song_audio(self.song_recording(phrase))
        empty = self.assess_from_library(recording, song_title="未收录歌曲")
        self.assertEqual(empty["reference_search_reason"], "title_not_found")
        self.upload_song_audio(self.song_recording([60, 62, 63, 66, 67, 64, 61]), "/reference")
        with (patch("deepvocal.audio.assess") as builtin,
              patch("deepvocal.assessment.measure_song") as measure,
              patch("deepvocal.practice_agent.prepare_evidence_practice") as plan):
            report = self.assess_from_library(recording, "测评并安排练习计划", song_title="song")
            builtin.assert_not_called()
            measure.assert_not_called()
            plan.assert_not_called()
        self.assertEqual(report["issue"], "reference_not_found")
        self.assertEqual(report["quality"], "requires_reference")
        self.assertEqual(report["recording_audio_id"], recording["id"])
        self.assertIn("上传对应片段的对比歌声", report["summary"])
        self.assertEqual(report["notes"], [])
        self.assertEqual([task["status"] for task in report["task_statuses"]], ["needs_input", "blocked"])
        self.assertEqual(report["assessment_agent"]["actions"], [
            {"tool": SEARCH, "status": "needs_input"}, {"tool": FINISH, "status": "needs_input"},
        ])
        store = self.client.app.state.store
        with store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_observations").fetchone()[0], 0)
        reference = self.upload_song_audio(self.song_recording(phrase), "/reference")
        retry = self.assess_song(reference, recording)["result"]
        self.assertEqual(retry["quality"], "usable")
        self.assertEqual(retry["recording_audio_id"], recording["id"])
        self.assertEqual([action["tool"] for action in retry["assessment_agent"]["actions"]], [ANALYZE, FINISH])

    def test_library_refuses_ambiguous_matches(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        reference_wav = self.song_recording(phrase + [52, 55, 58, 53] + phrase)
        self.upload_song_audio(reference_wav, "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase))
        report = self.assess_from_library(recording, song_title="song")
        self.assertEqual(report["issue"], "reference_not_found")
        self.assertNotIn("reference", report)

    def test_song_title_limits_fingerprint_search_to_the_named_song(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        expected = self.upload_song_audio(self.song_recording([48, 52] + phrase + [74]), "/reference",
                                          filename="晴天-歌手.wav")
        other = self.upload_song_audio(self.song_recording([50, 53] + phrase + [72]), "/reference",
                                       filename="other.wav", song_title="另一首")
        recording = self.upload_song_audio(self.song_recording(phrase))
        report = self.assess_from_library(recording, song_title="《晴 天》")
        self.assertEqual(report["reference"]["audio_id"], expected["id"])
        self.assertGreaterEqual(report["reference"]["fingerprint_hits"], 2)
        self.assertEqual(self.assess_from_library(recording, song_title="另一首")["reference"]["audio_id"], other["id"])

        missing = self.assess_from_library(recording, song_title="不存在的歌")
        self.assertEqual(missing["reference_search_reason"], "title_not_found")
        self.assertIn("请核对歌名", missing["summary"])
        self.assertNotIn("reference", missing)

    def test_assessment_asks_for_title_and_reuses_recording_after_chat_reply(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        expected = self.upload_song_audio(self.song_recording(phrase), "/reference", song_title="晴天")
        recording = self.upload_song_audio(self.song_recording(phrase))

        def choose(model, messages, *, tools):
            if messages[-1]["role"] == "user":
                task = json.loads(messages[-1]["content"])
                self.assertIsNone(task["song_title"])
                # The first reply deliberately proposes a title that the user did not mention.
                return assessment_tool(SEARCH, {"song_title": "晴天"})
            return assessment_model_output(model, messages)

        store = self.client.app.state.store
        with patch.object(store, "search_reference_motifs") as search:
            missing = self.assess_from_library(recording)
            self.assertEqual(missing["issue"], "song_title_required")
            self.assertEqual(missing["assessment_agent"]["actions"], [
                {"tool": SEARCH, "status": "needs_input"}, {"tool": FINISH, "status": "needs_input"},
            ])
            search.assert_not_called()

        with (patch("deepvocal.assessment_agent.model_step", side_effect=choose),
              patch.object(store, "search_reference_motifs", wraps=store.search_reference_motifs) as search):
            response = self.client.post("/api/chat", json={
                "intent": "assessment", "query": "请帮我看看音准", "audio_id": recording["id"], "alignment_mode": "signal",
            })
            self.assertEqual(response.status_code, 200, response.text)
            first = response.json()
            result = self.wait_for(lambda: self.status(first["job"]["id"]))["result"]
            self.assertEqual(result["issue"], "song_title_required")
            self.assertEqual(result["task_statuses"][0]["status"], "needs_input")
            self.assertIn("请在对话框里告诉我", result["summary"])
            self.assertEqual(result["recording_audio_id"], recording["id"])
            search.assert_not_called()

            retry = self.assess_from_library(recording, "晴天", intent="assessment", session_id=first["session_id"])
            self.assertEqual(retry["reference"]["audio_id"], expected["id"])
            self.assertEqual(retry["recording_audio_id"], recording["id"])
            self.assertEqual(search.call_args.kwargs["title_key"], "晴天")

    def test_named_song_without_fingerprint_hits_still_requires_audio_alignment(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        expected = self.upload_song_audio(self.song_recording(phrase), "/reference", song_title="测试歌曲")
        recording = self.upload_song_audio(self.song_recording(phrase))
        unrelated = self.upload_song_audio(self.song_recording([72, 50, 75, 49, 71, 46, 76]))
        with patch("deepvocal.assessment.melody_motifs", return_value=set()):
            report = self.assess_from_library(recording, song_title="测试歌曲")
            rejected = self.assess_from_library(unrelated, song_title="测试歌曲")
        self.assertEqual(report["reference"]["audio_id"], expected["id"])
        self.assertEqual(report["reference"]["fingerprint_hits"], 0)
        self.assertEqual(rejected["issue"], "reference_not_found")
        self.assertEqual(rejected["reference_search_reason"], "alignment_uncertain")

    def test_assessment_react_waits_for_search_and_scoring_before_finishing(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        self.upload_song_audio(self.song_recording([53, 55] + phrase + [72, 73]), "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase))
        replies = [
            assessment_tool(ANALYZE, call_id="before_search"),
            assessment_tool(SEARCH, {"song_title": "song"}, call_id="search"),
            assessment_tool(FINISH, {"status": "completed"}, "before_analysis"),
            assessment_tool(ANALYZE, call_id="analyze"),
            assessment_tool(FINISH, {"status": "completed"}, "finish"),
        ]
        store = self.client.app.state.store
        with (patch("deepvocal.assessment_agent.model_step", side_effect=replies),
              patch.object(store, "search_reference_motifs", wraps=store.search_reference_motifs) as search,
              patch("deepvocal.assessment.measure_song", wraps=measure_song) as measure):
            report = self.assess_from_library(recording, "我唱的是《song》，请测评。")
        search.assert_called_once()
        measure.assert_called_once()
        self.assertEqual([item["status"] for item in report["assessment_agent"]["actions"]],
                         ["rejected", "reference_found", "rejected", "completed", "completed"])

    def test_audio_llm_checks_attachment_match_before_assessment(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        reference = self.upload_song_audio(self.song_recording([53, 55, 53] + phrase + [71]), "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase))
        candidates = [{"index": 1, "start_seconds": 2, "end_seconds": 6.67}]
        model = Model("dashscope", "fixture-audio", "fixture")
        for first, second, issue in (("matched", "matched", "general"),
                                     ("matched", "mismatch", "reference_mismatch"),
                                     ("matched", "uncertain", "alignment_uncertain"),
                                     ("mismatch", None, "reference_mismatch")):
            with self.subTest(first=first, second=second):
                replies = [{"match": verdict, "candidates": candidates if verdict == "matched" else []}
                           for verdict in (first, second) if verdict is not None]
                responses = [{"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
                             for value in replies]
                with (patch("deepvocal.audio_alignment.audio_model", return_value=model),
                      patch.object(Model, "call", side_effect=responses) as api,
                      patch("deepvocal.audio_alignment.refine_with_lyrics", side_effect=lambda *args: args[-1]),
                      patch("deepvocal.assessment_agent.assess_aligned_audio", return_value=report_metrics("general")) as score):
                    report = self.assess_song(reference, recording, alignment_mode="llm")["result"]
                self.assertEqual(report["issue"], issue)
                self.assertEqual(api.call_count, len(replies))
                if issue == "general":
                    score.assert_called_once()
                    self.assertEqual(report["assessment_agent"]["status"], "completed")
                else:
                    score.assert_not_called()
                    self.assertEqual(report["assessment_agent"]["status"], "needs_input")
                    self.assertEqual(report["candidates"], [])
                    self.assertEqual(report["recording_audio_id"], recording["id"])
                    self.assertIn("请上传你演唱的那首歌" if issue == "reference_mismatch" else "暂未确认", report["summary"])

    def test_selected_reference_range_is_checked_before_scoring(self):
        recording = self.upload_song_audio(self.song_recording([60, 63, 67, 65, 69, 64, 61]))
        reference = self.upload_song_audio(self.song_recording([61, 60, 70, 65, 63, 68, 66]), "/reference")
        model = Model("dashscope", "fixture-audio", "fixture")
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"match":"mismatch"}'}}]}
        with (patch("deepvocal.audio_alignment.audio_model", return_value=model),
              patch.object(Model, "call", return_value=response) as api,
              patch("deepvocal.assessment_agent.assess_aligned_audio") as score):
            report = self.assess_song(reference, recording, alignment_mode="llm",
                                      reference_start_seconds=0, reference_end_seconds=3)["result"]
        api.assert_called_once()
        score.assert_not_called()
        self.assertEqual(report["issue"], "reference_segment_mismatch")
        self.assertEqual(report["assessment_agent"]["status"], "needs_input")
        self.assertNotIn("clip_audio_id", report["reference"])

    def test_attached_reference_uses_alignment_even_if_model_attempts_unavailable_search(self):
        wav = self.song_recording([60, 63, 67, 65, 69, 64, 61])
        reference = self.upload_song_audio(wav, "/reference")
        recording = self.upload_song_audio(wav)
        replies = iter([
            assessment_tool(SEARCH, call_id="unavailable_search"),
            assessment_tool(ALIGN, call_id="align_attachment"),
            assessment_tool(ANALYZE, call_id="analyze"),
            assessment_tool(FINISH, {"status": "completed"}, "finish"),
        ])

        def choose(model, messages, *, tools):
            names = {entry["function"]["name"] for entry in tools}
            self.assertNotIn(SEARCH, names)
            self.assertIn(ALIGN, names)
            if messages[-1].get("tool_call_id") == "unavailable_search":
                observation = json.loads(messages[-1]["content"])
                self.assertEqual(observation["status"], "rejected")
                self.assertIn("已有参考附件", observation["message"])
            return next(replies)

        store = self.client.app.state.store
        with (patch("deepvocal.assessment_agent.model_step", side_effect=choose),
              patch.object(store, "search_reference_motifs") as search,
              patch("deepvocal.assessment_agent.align_audio", return_value=({"audio_id": reference["id"]}, None)) as align,
              patch("deepvocal.assessment_agent.assess_aligned_audio", return_value=report_metrics("general")) as score):
            report = self.assess_song(reference, recording, alignment_mode="llm",
                                      query="我上传了参考，也请从曲库找参考")["result"]
        search.assert_not_called()
        self.assertEqual(align.call_args.args[2][0]["audio_id"], reference["id"])
        score.assert_called_once()
        self.assertEqual(report["assessment_agent"]["actions"], [
            {"tool": SEARCH, "status": "rejected"},
            {"tool": ALIGN, "status": "aligned"},
            {"tool": ANALYZE, "status": "completed"},
            {"tool": FINISH, "status": "completed"},
        ])

    def test_audio_llm_searches_full_song_and_scores_only_the_saved_original_slice(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        source = self.song_recording([53, 55, 53] + phrase + [74, 72, 70])
        reference = self.upload_song_audio(source, "/reference", song_title="测试全曲")
        recording = self.upload_song_audio(self.song_recording(phrase))
        proposed = {"candidates": [{"index": 1, "start_seconds": 2, "end_seconds": 6.67}]}
        feedback = {"assessable": True, "focus": "timing", "summary": "尾音稍晚。", "intonation": "旋律接近参考。",
                    "rhythm": "尾音稍晚。", "stability": "持续音平稳。", "exercise": "按参考分句跟唱。"}

        def response(path, payload, **options):
            items = payload["messages"][0]["content"]
            assert all(isinstance(item, dict) for item in items)
            media = [b64decode(item["input_audio"]["data"].split(",", 1)[1]) for item in items if item["type"] == "input_audio"]
            self.assertEqual(len(media), 2)
            song_duration = len(audio.read_wave(media[1])[0]) / audio.read_wave(media[1])[1]
            if "声乐导师" in items[0]["text"]:
                self.assertAlmostEqual(song_duration, 4.67, delta=.01)
                value = feedback
            else:
                self.assertGreater(song_duration, 6.67)
                value = ([{"candidate": 1, "start_seconds": 2, "end_seconds": 6.67}]
                         if "复听" in items[0]["text"] else [proposed])
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

        multimodal = Model("dashscope", "fixture-audio", "fixture")
        with (patch("deepvocal.audio_alignment.audio_model", return_value=multimodal),
              patch("deepvocal.audio_alignment.refine_with_lyrics", side_effect=lambda *args: args[-1]),
              patch("deepvocal.model.Model.call", side_effect=response) as api):
            report = self.assess_from_library(recording, song_title="测试全曲", alignment_mode="llm")
        self.assertEqual(api.call_count, 3)
        self.assertEqual([item["tool"] for item in report["assessment_agent"]["actions"]], [SEARCH, ALIGN, ANALYZE, FINISH])
        self.assertEqual(report["reference"]["audio_id"], reference["id"])
        self.assertEqual(report["audio_feedback"], feedback)
        self.assertIsNone(report["median_pitch_error_cents"])
        self.assertEqual(self.client.get(report["reference"]["clip_audio_url"]).content, cut_wave(source, 2, 6.67))

    def test_audio_llm_waits_for_repeated_passage_selection(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        ref = self.upload_song_audio(self.song_recording(phrase + phrase), "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase))
        with (patch("deepvocal.audio_alignment.audio_model", return_value=object()),
              patch("deepvocal.audio_alignment.locate_with_model", return_value=([(0, 4.6), (4.7, 9.3)], False)),
              patch("deepvocal.assessment_agent.assess_aligned_audio") as analyzer):
            report = self.assess_song(ref, recording, alignment_mode="llm")["result"]
            analyzer.assert_not_called()
        self.assertEqual(report["issue"], "alignment_uncertain")
        self.assertEqual(len(report["candidates"]), 2)
        self.assertEqual([item["tool"] for item in report["assessment_agent"]["actions"]], [ALIGN, FINISH])

    def test_plan_links_audio_feedback_to_drills_and_reuses_original_retest_segment(self):
        store = self.client.app.state.store
        with store.connect() as db:
            job_id = store.enqueue("assessment", {}, db)
            store.finish({"id": job_id, "kind": "assessment"}, report_metrics("pitch_flat") | {
                "assessment_method": "audio_model",
                "reference": {"title": "测试曲", "audio_id": "original", "segment_id": "phrase",
                              "segment_start_seconds": 12.5, "segment_end_seconds": 18.0, "transpose": 0},
                "audio_feedback": {"summary": "整体音准偏低。", "intonation": "句尾音高偏低。",
                                   "rhythm": "前半句没有拖拍，但最后一个字拖拍明显。", "stability": "持续音稳定。"},
            }, db=db)
        response = self.client.post("/api/chat", json={"intent": "practice_plan", "query": "按测评安排练习",
                                                       "assessment_job_id": job_id, "days": 3, "minutes": 10})
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()["plan"]
        self.assertEqual(plan["source_assessment"], job_id)
        self.assertEqual([item["id"] for item in plan["evidence"]], ["pitch_flat", "timing"])
        self.assertEqual(plan["evidence"][0]["location"], "句尾")
        self.assertEqual(plan["evidence"][1]["observation"], "最后一个字拖拍明显")
        self.assertEqual([day["stage"] for day in plan["sessions"]], ["分解练习", "连唱短句", "整段复测"])
        for day in plan["sessions"]:
            exercises = day["exercises"]
            self.assertEqual(sum(item["minutes"] for item in exercises), 10)
            self.assertEqual(exercises[1]["evidence_ids"], ["pitch_flat"])
            self.assertIn("句尾", exercises[1]["instructions"])
            self.assertEqual(exercises[2]["evidence_ids"], ["timing"])
        self.assertNotEqual(plan["sessions"][0]["exercises"], plan["sessions"][1]["exercises"])
        retest = plan["retest"]
        self.assertEqual(retest["reference"]["audio_id"], "original")
        self.assertEqual(retest["reference"]["segment_id"], "phrase")
        self.assertEqual(retest["reference"]["start_seconds"], 12.5)
        self.assertEqual(retest["reference"]["end_seconds"], 18.0)
        self.assertEqual([criterion["method"] for criterion in retest["criteria"]], ["listening", "listening"])
        self.assertTrue(all("baseline" not in criterion for criterion in retest["criteria"]))
        self.assertEqual(plan["generation"]["mode"], "rules")

        invalid = {"rationale": "坏计划测试", "minutes_per_day": 10, "sessions": [
            {"day": 1, "objective": "先练句尾", "exercises": [
                {"evidence_id": "unknown_skill", "step": 1, "minutes": 6, "instructions": "按测试方式练习"},
            ]},
        ]}
        with patch.object(Model, "chat_message", side_effect=lambda messages, **kw: planning_reply(messages, invalid)) as generator:
            planned = self.client.post("/api/chat", json={
                "intent": "practice_plan", "query": "按测评安排练习", "assessment_job_id": job_id,
                "days": 1, "minutes": 10, "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertEqual(generator.call_count, 3)
        fallback = planned.json()["plan"]
        self.assertEqual(fallback["generation"], {"mode": "rules", "reason": "invalid_model_plan",
                                                  "validation": {"check": "exercise_selection"}})
        self.assertEqual(fallback["retest"], retest)
        self.assertEqual(sum(item["minutes"] for item in fallback["sessions"][0]["exercises"]), 10)

    def test_automatic_practice_plan_uses_goal_and_rejects_repeated_stage(self):
        baseline = self.client.post("/api/chat", json={"intent": "practice_plan", "query": "帮我制定练习计划"})
        self.assertEqual(baseline.status_code, 200, baseline.text)
        self.assertEqual(baseline.json()["plan"]["days"], 1)
        self.assertEqual(baseline.json()["plan"]["evidence"][0]["source"], "baseline")

        proposal = {
            "rationale": "先找准拍点，再带回乐句。",
            "minutes_per_day": 9,
            "sessions": [
                {"day": day, "objective": "检查进拍", "exercises": [
                    {"evidence_id": "timing", "step": 1, "minutes": 7, "instructions": detail},
                ]} for day, detail in enumerate(("先标出歌词重拍。", "在起音前预数一拍。"), 1)
            ],
        }
        with patch.object(Model, "chat_message", side_effect=lambda messages, **kw: planning_reply(messages, proposal)):
            result = self.client.post("/api/chat", json={
                "intent": "practice_plan", "query": "帮我改善拖拍", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(result.status_code, 200, result.text)
        plan = result.json()["plan"]
        self.assertEqual(plan["generation"], {"mode": "rules", "reason": "invalid_model_plan",
                                              "validation": {"check": "exercise_progression"}})
        self.assertEqual(plan["focus"], "timing")
        self.assertEqual(plan["days"], 2)
        self.assertEqual(plan["evidence"][0]["source"], "goal")
        actions = [day["exercises"][1] for day in plan["sessions"]]
        self.assertEqual([action["step"] for action in actions], [1, 3])
        self.assertNotEqual(actions[0]["name"], actions[1]["name"])
        self.assertNotEqual(actions[0]["instructions"], actions[1]["instructions"])

    def test_practice_plan_keeps_valid_training_with_an_overlong_rationale(self):
        rationale = "这次先校准目标音。" + "在轻声唱准之后，把这个音放回原句，再录下来核对。" * 20
        proposal = {"rationale": rationale, "minutes_per_day": 8, "sessions": [
            {"day": 1, "objective": "校准句尾音高", "exercises": [
                {"evidence_id": "pitch_flat", "step": 1, "minutes": 6, "instructions": "每次先听清目标音再跟唱。"},
            ]},
        ]}
        with patch.object(Model, "chat_message", side_effect=lambda messages, **kw: planning_reply(messages, proposal)):
            result = self.client.post("/api/chat", json={
                "intent": "practice_plan", "query": "改善音高偏低", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(result.status_code, 200, result.text)
        plan = result.json()["plan"]
        self.assertEqual(plan["generation"], {"mode": "model", "model": "qwen-plus"})
        self.assertLessEqual(len(plan["rationale"]), 320)
        self.assertTrue(rationale.startswith(plan["rationale"]))
        self.assertTrue(plan["rationale"].endswith("。"))
        self.assertEqual(plan["sessions"][0]["exercises"][1]["minutes"], 6)

    def test_practice_plan_focus_and_budget_are_applied_to_model_sessions(self):
        store = self.client.app.state.store
        with store.connect() as db:
            job_id = store.enqueue("assessment", {}, db)
            store.finish({"id": job_id, "kind": "assessment"}, report_metrics("pitch_flat") | {
                "assessment_method": "audio_model",
                "reference": {"title": "测试练习曲", "segment_start_seconds": 20, "segment_end_seconds": 25},
                "audio_feedback": {"intonation": "整句音高偏低。", "stability": "持续音音高下坠。"},
            }, db=db)

        payload = {"intent": "practice_plan", "assessment_job_id": job_id,
                   "provider": "dashscope", "model": "qwen-plus"}
        headers = {"X-Model-API-Key": "mock"}

        def generate(issue):
            return {"rationale": "先分解重点，再复测完整乐句。", "minutes_per_day": 12, "sessions": [
                {"day": 1, "objective": "找准目标", "exercises": [
                    {"evidence_id": issue, "step": 1, "minutes": 12, "instructions": "先回听这一句再练。"},
                ]},
                {"day": 2, "objective": "回到完整乐句", "exercises": [
                    {"evidence_id": issue, "step": 2, "minutes": 12, "instructions": "先用同样的音量练好短句。"},
                ]},
            ]}

        for query, primary, evidence_ids, first_day_minutes in (
            ("练习长音稳定性", "stability", ["stability"], [2, 8, 2]),
            ("按测评制定练习计划", "pitch_flat", ["pitch_flat", "stability"], [2, 6, 2, 2]),
        ):
            def draft(context):
                self.assertEqual([item["evidence_id"] for item in context["approved_skills"]], evidence_ids)
                return generate(primary)

            with self.subTest(query=query), patch.object(Model, "chat_message",
                    side_effect=lambda messages, **kw: planning_reply(messages, draft)):
                response = self.client.post("/api/chat", json={**payload, "query": query}, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                plan = response.json()["plan"]
                self.assertEqual(plan["source_assessment"], job_id)
                self.assertEqual(plan["retest"]["reference"]["start_seconds"], 20)
                self.assertEqual(plan["generation"], {"mode": "model", "model": "qwen-plus"})
                self.assertEqual([row["id"] for row in plan["evidence"]], evidence_ids)
                self.assertEqual([row["evidence_id"] for row in plan["retest"]["criteria"]], evidence_ids)
                first, second = plan["sessions"]
                self.assertEqual(first["exercises"][-1]["evidence_ids"], evidence_ids)
                self.assertEqual([ex["minutes"] for ex in first["exercises"]], first_day_minutes)
                self.assertEqual([ex["minutes"] for ex in second["exercises"]], [2, 8, 2])
                self.assertEqual(second["exercises"][1]["step"], 2)
                self.assertIn("唱回完整乐句", second["exercises"][-1]["instructions"])
                self.assertIn("先回听这一句再练", first["exercises"][1]["instructions"])

    def test_planning_tools_find_qualitative_history_and_scope_current_report(self):
        store = self.client.app.state.store
        session = "planning-test"
        with store.connect() as db:
            report_id = store.enqueue("assessment", {}, db)
            store.finish({"id": report_id, "kind": "assessment"}, report_metrics("stability") | {
                "assessment_method": "audio_model",
                "reference": {"title": "山间微光-测试.wav", "audio_id": "reference", "segment_id": "chorus",
                              "segment_start_seconds": 18.5, "segment_end_seconds": 24.0},
                "audio_feedback": {"summary": "持续音有下坠。", "stability": "句尾持续音有下坠。"},
            }, db=db)
        store.record_turn(session, "测评这段录音", {"job": {"id": report_id, "kind": "assessment"}, "steps": ["assessment"]})
        self.assertIsNone(store.job(report_id)["result"]["median_pitch_error_cents"])

        def draft(context):
            self.assertEqual(context["reference"]["start_seconds"], 18.5)
            return {"rationale": "先练持续音，再回原句。", "minutes_per_day": 8, "sessions": [
                {"day": 1, "objective": "稳定尾音", "exercises": [
                    {"evidence_id": "stability", "step": 1, "minutes": 6, "instructions": "对照录音回听最后一个长音。"},
                ]},
            ]}

        payload = {"intent": "practice_plan", "provider": "dashscope", "model": "qwen-plus"}
        headers = {"X-Model-API-Key": "mock"}
        with patch.object(Model, "chat_message", side_effect=lambda messages, **kw: planning_reply(messages, draft)):
            for query, scope, session_id in (
                ("根据刚才《山间微光》的测评，只练长音稳定性，每天最多八分钟。", "linked", session),
                ("按上次《山间微光》的测评制定计划", "history", "new-session"),
            ):
                with self.subTest(scope=scope):
                    result = self.client.post("/api/chat", json={
                        **payload, "query": query, "session_id": session_id,
                        **({"assessment_job_id": "stale-session-report", "assessment_source": "session"}
                           if scope == "history" else {})}, headers=headers).json()
                    plan = result["plan"]
                    self.assertEqual(plan["source_assessment"], report_id)
                    self.assertEqual(plan["retest"]["reference"]["segment_id"], "chorus")
                    self.assertEqual(plan["retest"]["criteria"][0]["method"], "listening")
                    self.assertEqual([action["tool"] for action in result["planning_agent"]["actions"]],
                                     [SEARCH_REPORT, BUILD_PLAN, FINISH_PLAN])
            missing = self.client.post("/api/chat", json={
                **payload, "query": "根据刚才的测评制定计划", "session_id": "empty-session"}, headers=headers).json()
            self.assertNotIn("plan", missing)
            self.assertEqual(missing["task_statuses"][0]["status"], "needs_input")
            self.assertEqual([action["status"] for action in missing["planning_agent"]["actions"]],
                             ["needs_input", "needs_input"])

    def test_planning_tool_repairs_draft_and_keeps_spoken_time_cap_on_fallback(self):
        payload = {"intent": "practice_plan", "provider": "dashscope", "model": "qwen-plus",
                   "query": "只练长音稳定性，每天最多8分钟。"}

        for repair in (True, False):
            drafts = iter((12, 8 if repair else 12))

            def draft(context):
                self.assertEqual(context["schedule"]["minutes_per_day_range"], [5, 8])
                return {"rationale": "分解长音并复测。", "minutes_per_day": next(drafts), "sessions": [
                    {"day": 1, "objective": "稳定长音", "exercises": [
                        {"evidence_id": "stability", "step": 1, "minutes": 8, "instructions": "轻声唱完再回听。"},
                    ]},
                ]}

            with self.subTest(repair=repair), patch.object(Model, "chat_message",
                    side_effect=lambda messages, **kw: planning_reply(messages, draft)):
                result = self.client.post("/api/chat", json=payload, headers={"X-Model-API-Key": "mock"}).json()
                plan = result["plan"]
                self.assertEqual(plan["minutes_per_day"], 8)
                self.assertEqual(plan["generation"]["mode"], "model" if repair else "rules")
                self.assertEqual([item["status"] for item in result["planning_agent"]["actions"]],
                                 ["invalid_plan", "validated", "completed"] if repair else ["invalid_plan", "invalid_plan"])
                self.assertIsNone(plan["source_assessment"])

    def test_audio_alignment_refines_model_boundaries_from_word_timestamps(self):
        wav = self.song_recording([60, 63, 67, 65, 69, 64, 61] * 3)
        ref = self.upload_song_audio(wav, "/reference")
        phrase = "春天来了我唱歌"
        query_words = [(char, i * 400, (i + 1) * 400) for i, char in enumerate(phrase)]
        reference_words = [(char, 1800 + i * 400, 1800 + (i + 1) * 400) for i, char in enumerate(phrase)]
        selected = {"audio_id": ref["id"], "start_seconds": 5, "end_seconds": 8,
                    "duration_seconds": ref["duration_seconds"]}
        recorded, rate = audio.read_wave(wav)
        store = self.client.app.state.store
        with patch("deepvocal.audio_alignment.timed_characters", side_effect=[query_words, reference_words]):
            match = refine_with_lyrics(store, object(), recorded, rate, selected)
        self.assertEqual((match["start_seconds"], match["end_seconds"], match["boundary_source"]),
                         (2.8, 5.6, "word_timestamps"))

    def test_dtw_backup_finds_a_subsequence_with_an_overall_pitch_shift(self):
        phrase = np.repeat([60, 63, 67, 65, 69], 5).astype(float)
        original = np.r_[np.tile([50, 55], 10), phrase, np.tile([74, 75], 9)]
        first = subsequence_candidates(original, phrase + 2)[0]
        self.assertAlmostEqual(first[0], 20 * .08, delta=.08)
        self.assertAlmostEqual(first[1], 45 * .08, delta=.08)

    def test_song_assessment_locates_clip_in_long_reference_and_recalls_repeat(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        prefix = [52, 56, 54, 57, 52] * 20
        reference = self.upload_song_audio(self.song_recording(prefix + phrase + [72, 74, 72]), "/reference")
        self.assertGreater(reference["duration_seconds"], 60)
        clip = self.upload_song_audio(self.song_recording(phrase, offset=0.4))
        report = self.assess_song(reference, clip, offset_seconds=0.4)["result"]
        self.assertEqual(report["reference"]["selection"], "automatic")
        self.assertAlmostEqual(report["reference"]["start_seconds"], len(prefix) * (60 / 90), delta=0.16)
        self.assertAlmostEqual(report["alignment"]["recording_start_seconds"], 0.4, delta=0.12)
        self.assertGreaterEqual(report["matched_notes"], 6)
        self.assertIn("reference_start_seconds", report["notes"][0])
        self.assertLess(abs(report["median_pitch_error_cents"]), 15)

        start = round(len(prefix) * 60 / 90 - 0.12, 3)
        end = round((len(prefix) + len(phrase)) * 60 / 90 + 0.12, 3)
        limits = {"reference_start_seconds": start, "reference_end_seconds": end}
        flat = self.upload_song_audio(self.song_recording(phrase, transpose=-1))
        first = self.assess_song(reference, flat, **limits)
        self.assertEqual(first["result"]["issue"], "pitch_flat")
        second = self.assess_song(reference, clip, offset_seconds=0.4, query="测评并安排练习计划", **limits)
        result = second["result"]
        self.assertEqual(result["reference"]["selection"], "manual")
        self.assertEqual(result["reference"]["segment_id"], first["result"]["reference"]["segment_id"])
        self.assertEqual(result["comparison"]["previous"]["report_id"], first["id"])
        self.assertAlmostEqual(result["comparison"]["absolute_bias_reduction_cents"], 100, delta=15)
        self.assertIn("plan", result)
        history = self.client.get("/api/memory/assessments", params={"segment_id": result["reference"]["segment_id"]}).json()
        self.assertEqual([row["report_id"] for row in history["observations"]], [second["id"], first["id"]])

    def test_repeated_phrase_requires_selection_before_scoring(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        middle = [48, 50, 52, 53, 50]
        reference = self.upload_song_audio(self.song_recording(phrase + middle + phrase + middle), "/reference")
        recording = self.upload_song_audio(self.song_recording(phrase))
        report = self.assess_song(reference, recording, query="测评并安排练习计划")["result"]
        self.assertEqual(report["issue"], "alignment_uncertain")
        self.assertGreaterEqual(len(report["candidates"]), 2)
        self.assertIn("plan_error", report)
        self.assertEqual(report["notes"], [])
        self.assertEqual(report["quality"], "needs_selection")
        self.assertEqual([step["status"] for step in report["task_statuses"]], ["needs_input", "blocked"])
        self.assertEqual(report["recording_audio_id"], recording["id"])
        chosen = max(report["candidates"], key=lambda candidate: candidate["start_seconds"])
        selected = self.assess_song(
            reference, recording, reference_start_seconds=chosen["start_seconds"], reference_end_seconds=chosen["end_seconds"]
        )["result"]
        self.assertIn("segment_id", selected["reference"])
        self.assertGreater(selected["reference"]["start_seconds"], 7)
        self.assertGreaterEqual(selected["matched_notes"], 6)

    def test_song_assessment_tracks_speed_and_respects_requested_transposition(self):
        phrase = [60, 63, 67, 65, 69, 64, 61]
        reference = self.upload_song_audio(self.song_recording([52, 56, 52] + phrase + [74, 73, 72]), "/reference")
        clip = self.upload_song_audio(self.song_recording(phrase, bpm=99, transpose=2))
        report = self.assess_song(reference, clip, transpose=2)["result"]
        self.assertEqual(report["quality"], "usable")
        self.assertAlmostEqual(report["alignment"]["relative_speed"], 1.1, delta=0.03)
        self.assertLess(abs(report["median_pitch_error_cents"]), 15)

        quiet = self.upload_song_audio(audio.to_wave(np.zeros(16000), 8000), "/reference")
        missing = self.assess_song(quiet, clip)["result"]
        self.assertEqual(missing["issue"], "reference_unusable")
        self.assertIsNone(missing["median_pitch_error_cents"])

    def test_song_assessment_validates_reference_bounds_and_audio_kind(self):
        wav = self.song_recording([60, 62, 64])
        recording = self.upload_song_audio(wav)
        request = {"query": "测评", "intent": "assessment", "audio_id": recording["id"],
                   "reference_audio_id": recording["id"], "alignment_mode": "signal"}
        self.assertEqual(self.client.post("/api/chat", json={**request, "reference_start_seconds": 0}).status_code, 422)
        wrong_kind = self.client.post("/api/chat", json=request).json()
        self.assertEqual(wrong_kind["task_statuses"][0]["status"], "failed")
        reference = self.upload_song_audio(wav, "/reference")
        result = self.client.post("/api/chat", json={**request, "reference_audio_id": reference["id"],
                                                    "reference_start_seconds": 0, "reference_end_seconds": 9}).json()
        job = self.wait_for(lambda: self.status(result["job"]["id"]))
        self.assertEqual(job["state"], "failed")
        self.assertIn("超出音频长度", job["error"])

    def test_song_timing_uses_clear_onsets_without_reporting_legato_transitions(self):
        reference, performance = np.full(48, np.nan), np.full(48, np.nan)
        for start, end, midi, delay in [(4, 13, 60, 0), (16, 25, 64, 3), (29, 38, 67, 0), (38, 45, 65, 0)]:
            reference[start:end] = midi
            performance[start + delay:end] = midi
        report = measure_song({"transpose": 0}, reference, performance,
                              {"start": 4, "end": 45, "origin": 4, "scale": 1}, 1.92)
        self.assertEqual(report["timing_observations"], 3)
        self.assertEqual([note["onset_error_ms"] for note in report["notes"]], [0, 120, 0, None])

    def test_assessment_react_rejects_unsupported_actions_and_reports_to_controller(self):
        recording = self.upload_song_audio(self.song_recording(audio.MELODIES["warmup"]["notes"]))
        req = {"query": "测评新录音", "intent": "assessment", "audio_id": recording["id"], "reference_mode": "builtin"}
        replies = [
            assessment_tool(FINISH, {"status": "completed"}, "premature"),
            assessment_tool(ANALYZE, {"audio_id": "unapproved"}, "bad_argument"),
            assessment_tool(ANALYZE, call_id="analyze"),
            assessment_tool(FINISH, {"status": "needs_input"}, "wrong_status"),
            assessment_tool(FINISH, {"status": "completed"}, "verified"),
        ]
        with (patch("deepvocal.assessment_agent.model_step", side_effect=replies) as decide,
              patch("deepvocal.audio.assess", wraps=audio.assess) as audio_tool,
              patch("deepvocal.main.resume_assessment", wraps=resume_assessment) as controller):
            submission = self.client.post("/api/chat", json=req).json()
            job = self.wait_for(lambda: self.status(submission["job"]["id"]))
            audio_tool.assert_called_once()
            controller.assert_called_once()
        self.assertEqual(job["state"], "succeeded")
        observed = json.loads(decide.call_args_list[-1].args[1][-1]["content"])
        self.assertEqual(observed, {"status": "rejected", "message": "验收状态必须与实际工具结果一致。",
                                    "expected_status": "completed"})
        self.assertEqual(job["result"]["assessment_agent"]["turns"], MAX_TURNS)
        actions = job["result"]["assessment_agent"]["actions"]
        self.assertEqual([action["status"] for action in actions],
                         ["rejected", "rejected", "completed", "rejected", "completed"])
        self.assertEqual(job["result"]["task_statuses"][0]["status"], "completed")

    def test_assessment_react_reuses_existing_result_without_rescoring_or_indexing(self):
        recording = self.upload_song_audio(self.song_recording(audio.MELODIES["warmup"]["notes"]))
        submission = self.client.post("/api/chat", json={
            "query": "测评录音", "intent": "assessment", "audio_id": recording["id"], "reference_mode": "builtin",
        }).json()
        previous = self.wait_for(lambda: self.status(submission["job"]["id"]))
        with patch("deepvocal.audio.assess") as audio_tool:
            reuse = self.client.post("/api/chat", json={
                "query": "看一下上次的测评报告", "intent": "assessment", "assessment_job_id": previous["id"],
            }).json()
            job = self.wait_for(lambda: self.status(reuse["job"]["id"]))
            audio_tool.assert_not_called()
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["result"]["source_assessment"], previous["id"])
        self.assertEqual([item["tool"] for item in job["result"]["assessment_agent"]["actions"]], [REUSE, FINISH])
        history = self.client.get("/api/memory/assessments?melody_id=warmup").json()
        self.assertEqual([row["report_id"] for row in history["observations"]], [previous["id"]])

    def test_assessment_react_model_failure_blocks_dependent_plan(self):
        recording = self.upload_song_audio(self.song_recording(audio.MELODIES["warmup"]["notes"]))
        with (patch("deepvocal.assessment_agent.model_step", return_value={"role": "assistant", "content": "已经测完"}) as decide,
              patch("deepvocal.audio.assess") as audio_tool,
              patch("deepvocal.practice_agent.prepare_evidence_practice") as planner):
            submission = self.client.post("/api/chat", json={
                "query": "测评并安排练习", "audio_id": recording["id"],
            }).json()
            job = self.wait_for(lambda: self.status(submission["job"]["id"]))
            audio_tool.assert_not_called()
            planner.assert_not_called()
        self.assertEqual(decide.call_count, MAX_TURNS)
        self.assertEqual(job["state"], "failed")
        self.assertEqual([step["status"] for step in job["result"]["task_statuses"]], ["failed", "blocked"])

    def test_assessment_requires_a_tool_calling_model(self):
        recording = self.upload_song_audio(self.song_recording(audio.MELODIES["warmup"]["notes"]))
        with patch("deepvocal.main.assessment_model_for", wraps=assessment_model_for):
            result = self.client.post("/api/chat", json={
                "query": "测评录音", "intent": "assessment", "audio_id": recording["id"],
            }).json()
        self.assertEqual(result["task_statuses"][0]["status"], "failed")
        self.assertIn("需要支持工具调用的模型", result["answer"])
        self.assertNotIn("job", result)

    def test_knowledge_and_guide_generation(self):
        qa = self.qa("什么是换声区？")
        self.assertEqual(qa["evidence"][0]["id"], "register")
        self.assertTrue(any(d["path"] for d in qa["evidence"]))
        fake = self.client.post("/api/audio", files={"file": ("bad.wav", b"bad", "audio/wav")})
        self.assertEqual(fake.status_code, 422)
        chat = self.client.post("/api/chat", json={"query": "生成示范音频", "intent": "singing"}).json()
        job = self.wait_for(lambda: self.status(chat["job"]["id"]))
        self.assertEqual(job["result"]["type"], "guide")
        self.assertTrue(self.client.get(job["result"]["audio_url"]).content.startswith(b"RIFF"))

    def test_knowledge_answer_and_recording_assessment_run_concurrently(self):
        melody = self.client.get("/api/melodies/warmup/guide.wav")
        recording = self.client.post("/api/audio", files={"file": ("guide.wav", melody.content, "audio/wav")})
        self.assertEqual(recording.status_code, 201)
        overlap = Barrier(2, timeout=5)
        original_assess = audio.assess

        def assess(data, spec):
            overlap.wait()
            return original_assess(data, spec)

        def answer(messages, *, tool_choice, **kwargs):
            question = json.loads(messages[1]["content"])["question"]
            self.assertEqual(question, "什么是换声区")
            if tool_choice == "auto":
                return tool_reply(question)
            overlap.wait()
            return {"content": "换声区是声区转换较明显的一段音域。[register]"}

        with patch("deepvocal.audio.assess", side_effect=assess), patch("deepvocal.model.Model.chat_message", side_effect=answer):
            response = self.client.post("/api/chat", json={
                "query": "什么是换声区？我这次唱得对吗？", "intent": "auto", "audio_id": recording.json()["id"],
                "reference_mode": "builtin", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["steps"], ["qa", "assessment"])
            self.assertEqual([item["id"] for item in result["evidence"]], ["register"])
            self.assertIn("录音已进入测评队列", result["answer"])
            report = self.wait_for(lambda: self.status(result["job"]["id"]))
            self.assertEqual(report["state"], "succeeded", report)
            self.assertEqual(report["result"]["matched_notes"], 8)
        turns = self.client.get(f"/api/sessions/{result['session_id']}/turns").json()["turns"]
        self.assertEqual(turns[0]["result"]["steps"], ["qa", "assessment"])
        self.assertEqual(turns[0]["job"]["id"], result["job"]["id"])

    def test_explicit_combined_intent_can_answer_while_audio_is_missing(self):
        result = self.qa("什么是换声区？", intent="qa_assessment")
        self.assertEqual(result["steps"], ["qa", "assessment"])
        self.assertIn("请先上传 WAV 或 M4A 清唱录音", result["answer"])
        self.assertEqual(result["evidence"][0]["id"], "register")
        self.assertNotIn("job", result)

    def test_controller_dispatches_independent_agents_and_resumes_after_assessment(self):
        melody = self.client.get("/api/melodies/warmup/guide.wav?transpose=-1")
        recording = self.client.post("/api/audio", files={"file": ("guide.wav", melody.content, "audio/wav")}).json()
        tasks = [
            {"kind": "qa", "query": "什么是换声区？", "depends_on": []},
            {"kind": "assessment", "query": "测评录音", "depends_on": []},
            {"kind": "practice_plan", "query": "根据测评结果制定练习计划", "depends_on": ["assessment"]},
            {"kind": "singing", "query": "生成示范音频", "depends_on": []},
        ]
        overlap, assessed = Barrier(3, timeout=7), Event()
        assess, generate = audio.assess, audio.generate

        def run_assessment(data, spec):
            overlap.wait()
            report = assess(data, spec)
            assessed.set()
            return report

        def run_singing(spec):
            overlap.wait()
            return generate(spec)

        def draft(context):
            self.assertEqual(context["primary_evidence_id"], "pitch_flat")
            self.assertEqual(context["evidence"][0]["source"], "assessment")
            self.assertLess(context["retest"][0]["baseline"]["value"], -90)
            self.assertEqual(context["schedule"]["suggested_days"], 2)
            return {
                    "rationale": "音高偏低是声带张力不足造成的。",
                    "minutes_per_day": 9,
                    "sessions": [{
                        "day": day, "objective": objective,
                        "exercises": [{"evidence_id": "pitch_flat", "minutes": 7, "step": day,
                                       "instructions": objective + "。"}],
                    } for day, objective in enumerate(("听标准音并轻声模唱", "连唱目标音前后的短句", "唱完整练习并比较音高"), 1)],
            }

        def reply(messages, *, tool_choice=None, **kwargs):
            if "练习规划师" in messages[0]["content"]:
                self.assertTrue(assessed.is_set())
                return planning_reply(messages, draft)
            if tool_choice is None:
                return {"content": json.dumps({"tasks": list(reversed(tasks))}, ensure_ascii=False)}
            self.assertEqual(json.loads(messages[1]["content"])["question"], "什么是换声区？")
            if tool_choice == "auto":
                return tool_reply("什么是换声区？")
            overlap.wait()
            return {"content": "换声区是声区过渡的音域。[register]"}

        with (patch("deepvocal.audio.assess", side_effect=run_assessment),
              patch("deepvocal.audio.generate", side_effect=run_singing),
              patch("deepvocal.model.Model.chat_message", side_effect=reply)):
            response = self.client.post("/api/chat", json={
                "query": "什么是换声区？测评录音，再根据测评结果制定练习计划，同时生成示范音频。",
                "intent": "auto", "audio_id": recording["id"], "reference_mode": "builtin",
                "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["steps"], [task["kind"] for task in tasks])
            self.assertEqual(result["task_statuses"], [
                {"kind": "qa", "depends_on": [], "status": "completed"},
                {"kind": "assessment", "depends_on": [], "status": "queued"},
                {"kind": "practice_plan", "depends_on": ["assessment"], "status": "waiting"},
                {"kind": "singing", "depends_on": [], "status": "queued"},
            ])
            self.assertIn("[register]", result["answer"])
            self.assertEqual([job["kind"] for job in result["jobs"]], ["assessment", "singing"])
            self.assertEqual(result["job"], result["jobs"][0])
            assessment, singing = [self.wait_for(lambda: self.status(job["id"])) for job in result["jobs"]]
            self.assertEqual(assessment["state"], "succeeded", assessment)
            plan = assessment["result"]["plan"]
            self.assertEqual(plan["focus"], "pitch_flat")
            self.assertEqual(plan["generation"], {"mode": "model", "model": "qwen-plus"})
            self.assertEqual((plan["days"], plan["minutes_per_day"]), (3, 9))
            self.assertTrue(all(sum(item["minutes"] for item in day["exercises"]) == 9 for day in plan["sessions"]))
            self.assertIn("听标准音并轻声模唱", plan["rationale"])
            self.assertNotIn("声带张力", plan["rationale"])
            self.assertIn("连唱目标音前后的短句。", plan["sessions"][1]["exercises"][1]["instructions"])
            self.assertEqual(plan["retest"]["criteria"][0]["baseline"]["value"],
                             assessment["result"]["median_pitch_error_cents"])
            self.assertEqual(singing["state"], "succeeded", singing)
            self.assertEqual(assessment["result"]["planning_agent"]["actions"], [
                {"tool": SEARCH_REPORT, "status": "found"}, {"tool": BUILD_PLAN, "status": "validated"},
                {"tool": FINISH_PLAN, "status": "completed"},
            ])
        turn = self.client.get(f"/api/sessions/{result['session_id']}/turns").json()["turns"][0]
        self.assertEqual(turn["jobs"], [assessment, singing])

    def test_controller_blocks_plan_if_required_assessment_fails(self):
        with patch("deepvocal.practice_agent.prepare_evidence_practice") as planner:
            result = self.client.post("/api/chat", json={"query": "测评并安排练习", "intent": "auto"}).json()
        self.assertEqual([task["status"] for task in result["task_statuses"]], ["failed", "blocked"])
        self.assertNotIn("plan", result)
        planner.assert_not_called()

    def test_model_decision_failure_does_not_trigger_rag(self):
        retriever = self.client.app.state.retriever
        with (patch("deepvocal.model.Model.chat_message", side_effect=ModelError("模型暂不可用。")),
              patch.object(retriever, "search") as search):
            response = self.client.post("/api/chat", json={
                "query": "换声区是什么？", "intent": "qa", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertIn("模型暂不可用。", result["answer"])
            self.assertEqual(result["qa_generation"], {"mode": "model_error"})
            self.assertEqual(result["rag"], {"used": False})
            search.assert_not_called()

    def test_qa_requires_model_and_only_executes_supported_tool_calls(self):
        retriever = self.client.app.state.retriever
        body = {"query": "你好！", "intent": "qa", "retrieval": "graph"}
        with patch.object(retriever, "search") as search:
            response = self.client.post("/api/chat", json=body).json()
            self.assertEqual(response["qa_generation"], {"mode": "model_required"})
            self.assertEqual(response["rag"], {"used": False})
            for reply, mode in [({"content": "你好！"}, "model"),
                                (tool_reply("换声区", name="other_tool"), "model_error"),
                                (tool_reply("  "), "model_error")]:
                with self.subTest(reply=reply), patch("deepvocal.model.Model.chat_message", return_value=reply) as llm:
                    response = self.client.post("/api/chat", json={**body, "provider": "dashscope", "model": "qwen-plus"},
                                                headers={"X-Model-API-Key": "fixture-key"}).json()
                    self.assertEqual(response["qa_generation"]["mode"], mode)
                    self.assertEqual(response["rag"], {"used": False})
                    self.assertEqual(response["evidence"], [])
                    llm.assert_called_once()
                    self.assertEqual(llm.call_args.kwargs["tool_choice"], "auto")
            search.assert_not_called()

    def test_qa_shows_cited_evidence_or_regenerates_from_general_knowledge(self):
        headers = {"X-Model-API-Key": "mock"}
        query = {"query": "什么是换声区？", "intent": "qa", "provider": "dashscope", "model": "qwen-plus"}
        for reply, expected_ids in [("换声区是一段音域。[register]", ["register"]),
                                    ("NO_EVIDENCE", []), ("无依据答案。[unknown]", [])]:
            replies = [tool_reply(query["query"]), {"content": reply}]
            if not expected_ids:
                replies.append({"content": "声区交接可能出现音色变化。[unknown]"})
            with self.subTest(reply=reply), patch("deepvocal.model.Model.chat_message", side_effect=replies) as llm:
                response = self.client.post("/api/chat", json=query, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual([item["id"] for item in result["evidence"]], expected_ids)
                self.assertEqual(result["rag"], {"used": True, "tool": "search_knowledge", "query": query["query"], "retrieval": "graph"})
                choices = [call.kwargs.get("tool_choice") for call in llm.call_args_list]
                self.assertEqual(choices, ["auto", "none"] if expected_ids else ["auto", "none", None])
                self.assertEqual(llm.call_args_list[1].args[0][-1]["tool_call_id"], "call_search")
                if not expected_ids:
                    self.assertEqual(result["qa_generation"]["mode"], "model_fallback")
                    self.assertIn("知识库未找到", result["answer"])
                    self.assertIn("声区交接可能出现音色变化", result["answer"])
                    self.assertNotIn("[unknown]", result["answer"])
                    self.assertEqual([m["role"] for m in llm.call_args.args[0]], ["system", "user"])

    def test_chat_stream_assembles_tool_arguments_and_persists_final_answer(self):
        def upstream(*parts):
            frames = [{"choices": [{"delta": delta, "finish_reason": finish}]} for delta, finish in parts]
            text = "".join("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n" for frame in frames)
            return nullcontext(httpx.Response(200, text=text + "data: [DONE]\n\n"))

        start_tool = {"index": 0, "id": "call_search", "type": "function",
                      "function": {"name": "search_knowledge", "arguments": '{"query":'}}
        finish_tool = {"index": 0, "id": "", "function": {"arguments": '"换声区是什么"}'}}
        decision = upstream(({"tool_calls": [start_tool]}, None), ({"tool_calls": [finish_tool]}, None), ({}, "tool_calls"))
        answer = upstream(({"reasoning_content": "内部过程"}, None), ({"content": "换声区是"}, None),
                          ({"content": "声区切换的过渡区域。[ register ]"}, None), ({}, "stop"))
        with patch("deepvocal.model.httpx.stream", side_effect=[decision, answer]) as outbound:
            response = self.client.post("/api/chat/stream", json={
                "query": "什么是换声区？", "intent": "qa", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertNotIn("内部过程", response.text)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([event["type"] for event in events], ["status", "status", "status", "delta", "delta", "done"])
        self.assertEqual([event["phase"] for event in events[:3]], ["routing", "retrieving", "answering"])
        self.assertEqual(events[3]["text"], "换声区是")
        self.assertEqual([call.kwargs["json"]["tool_choice"] for call in outbound.call_args_list], ["auto", "none"])
        self.assertTrue(all(call.kwargs["json"]["stream"] for call in outbound.call_args_list))
        result = events[-1]["result"]
        self.assertEqual(result["rag"]["query"], "换声区是什么")
        self.assertEqual(result["answer"], "换声区是声区切换的过渡区域。[register]")
        self.assertEqual([item["id"] for item in result["evidence"]], ["register"])
        turns = self.client.get(f"/api/sessions/{result['session_id']}/turns").json()["turns"]
        self.assertEqual(turns[0]["result"]["answer"], result["answer"])

    def test_chat_stream_with_empty_retrieval_emits_general_knowledge_fallback(self):
        def respond(messages, *, on_text, tool_choice=None, **options):
            if tool_choice == "auto":
                return tool_reply("陌生曲目的唱法")
            self.assertIsNone(tool_choice)
            self.assertNotIn("tools", options)
            self.assertEqual([message["role"] for message in messages], ["system", "user"])
            on_text("先分句")
            on_text("练习节奏。")
            return {"content": "先分句练习节奏。"}

        with (patch.object(self.client.app.state.retriever, "search", return_value=[]),
              patch("deepvocal.model.Model.stream_message", side_effect=respond) as chat):
            response = self.client.post("/api/chat/stream", json={
                "query": "陌生曲目怎么唱？", "intent": "qa", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([event["phase"] for event in events if event["type"] == "status"], ["routing", "retrieving", "fallback"])
        self.assertEqual([event["text"] for event in events if event["type"] == "delta"], ["先分句", "练习节奏。"])
        result = events[-1]["result"]
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(result["qa_generation"]["mode"], "model_fallback")
        self.assertEqual(result["evidence"], [])
        self.assertTrue(result["answer"].startswith(events[2]["notice"]))
        self.assertEqual(chat.call_count, 2)

    def test_saved_sessions_and_paged_turns_with_job_results(self):
        first = self.qa("什么是换声区？")
        self.qa("另一个会话")
        session = first["session_id"]
        following = self.client.post("/api/chat", json={
            "session_id": session, "query": "生成参考音", "intent": "singing",
        }).json()
        self.wait_for(lambda: self.status(following["job"]["id"]))

        sidebar = self.client.get("/api/sessions?limit=1").json()
        self.assertEqual(sidebar["next_offset"], 1)
        self.assertEqual(sidebar["sessions"][0]["title"], "什么是换声区？")
        self.assertEqual(sidebar["sessions"][0]["turn_count"], 2)
        other = self.client.get("/api/sessions?limit=1&offset=1").json()
        self.assertEqual(other["sessions"][0]["title"], "另一个会话")
        self.assertIsNone(other["next_offset"])

        url = f"/api/sessions/{session}/turns"
        latest = self.client.get(url, params={"limit": 1}).json()
        self.assertEqual(latest["turns"][0]["id"], following["turn_id"])
        self.assertEqual(latest["turns"][0]["job"]["state"], "succeeded")
        self.assertEqual(latest["turns"][0]["job"]["result"]["type"], "guide")
        older = self.client.get(url, params={"limit": 1, "before": latest["next_before"]}).json()
        self.assertEqual(older["turns"][0]["id"], first["turn_id"])
        self.assertEqual(older["turns"][0]["result"]["evidence"][0]["id"], "register")
        self.assertIsNone(older["next_before"])
        self.assertEqual(self.client.get("/api/sessions/missing/turns").status_code, 404)

    def test_delete_session_removes_all_turns_and_their_question_memory(self):
        first = self.qa("什么是换声区？")
        session = first["session_id"]
        second = self.qa("音分是什么？", session_id=session)
        other = self.qa("什么是拖拍？")
        removed = {first["turn_id"], second["turn_id"]}
        self.wait_for(lambda: removed <= {fact["source"] for fact in self.client.get("/api/memory").json()["facts"]})

        url = f"/api/sessions/{session}"
        for _ in range(2):
            response = self.client.delete(url)
            self.assertEqual(response.status_code, 204)
            self.assertEqual(response.content, b"")
        self.assertEqual(self.client.get(url + "/turns").status_code, 404)
        self.assertEqual([s["id"] for s in self.client.get("/api/sessions").json()["sessions"]], [other["session_id"]])
        store = self.client.app.state.store
        self.assertEqual(store.history(session), [])
        store.store_fact({"category": "question", "subject": "register", "value": {"question": "重试"},
                          "source": first["turn_id"], "observed": time.time()})
        self.assertFalse(removed & {fact["source"] for fact in store.facts("question")})
        response = self.client.post("/api/chat", json={"session_id": session, "query": "继续", "intent": "qa"})
        self.assertEqual(response.status_code, 410)

    def test_delete_session_wins_over_a_chat_already_in_progress(self):
        first = self.qa("什么是换声区？")
        session = first["session_id"]
        app = self.client.app

        def finish_after_deletion(_):
            app.state.store.delete_session(session)
            return {"steps": ["qa"], "answer": "稍后完成的回答"}

        with patch.object(app.state.workflow, "invoke", side_effect=finish_after_deletion):
            response = self.client.post("/api/chat", json={"session_id": session, "query": "继续", "intent": "qa"})
        self.assertEqual(response.status_code, 410)
        self.assertEqual(app.state.store.history(session), [])

    def test_configured_model_header(self):
        response = httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        with patch("deepvocal.model.httpx.post", return_value=response) as outbound:
            check = self.client.post("/api/model/test", headers={"X-Model-API-Key": "test-key"},
                                     json={"provider": "deepseek", "model": "example-model"})
        self.assertEqual(check.status_code, 200)
        self.assertEqual(outbound.call_args.args[0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(outbound.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertNotIn("test-key", check.text)
        self.assertEqual(self.client.post("/api/model/test", json={"model": "example-model"}).status_code, 422)

    def test_dashscope_model_connection_and_chat(self):
        response = httpx.Response(200, json={"choices": [{"message": {"content": "百炼测试回答 [register]"}}]})
        body = {"provider": "dashscope", "model": "qwen-plus"}
        headers = {"X-Model-API-Key": "fixture-key"}
        with patch("deepvocal.model.httpx.post", return_value=response) as outbound:
            status = self.client.post("/api/model/test", headers=headers, json=body)
            self.assertEqual(status.json(), {"status": "ok", **body})
            self.assertEqual(outbound.call_args.args[0],
                             "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            self.assertIs(outbound.call_args.kwargs["trust_env"], False)
            workspace = "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            with patch.dict(os.environ, {"DEEPVOCAL_DASHSCOPE_API_BASE": workspace,
                                         "DEEPVOCAL_DASHSCOPE_USE_PROXY": "1"}):
                chat = self.client.post("/api/chat", headers=headers, json={
                    **body, "intent": "qa", "query": "什么是换声区？",
                })
            self.assertEqual(chat.json()["answer"], "百炼测试回答 [register]")
            self.assertEqual(outbound.call_args.args[0], workspace + "/chat/completions")
            self.assertIs(outbound.call_args.kwargs["trust_env"], True)
            self.assertEqual(outbound.call_args.kwargs["headers"]["Authorization"], "Bearer fixture-key")
            self.assertEqual(outbound.call_args.kwargs["json"]["model"], "qwen-plus")

    def test_dashscope_distinguishes_authentication_and_transport_errors(self):
        body = {"provider": "dashscope", "model": "qwen-plus"}
        headers = {"X-Model-API-Key": "fixture-key"}
        with patch("deepvocal.model.httpx.post", return_value=httpx.Response(401)):
            response = self.client.post("/api/model/test", headers=headers, json=body)
            self.assertEqual(response.status_code, 502)
            self.assertIn("HTTP 401", response.json()["detail"])
            self.assertIn("地域", response.json()["detail"])
        with patch("deepvocal.model.httpx.post", side_effect=httpx.ConnectTimeout("transport detail")):
            response = self.client.post("/api/model/test", headers=headers, json=body)
            self.assertIn("请求超时", response.json()["detail"])
            self.assertNotIn("transport detail", response.text)

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI is required")
    def test_saved_model_survives_restart_without_exposing_key(self):
        body = {"provider": "dashscope", "model": "qwen-plus"}
        fixture_key = "deepvocal-fixture-only-key"
        response = httpx.Response(200, json={"choices": [{"message": {"content": "已保存模型的回答 [register]"}}]})
        with TemporaryDirectory() as path:
            with TestClient(create_app(path)) as first:
                saved = first.put("/api/model/settings", json=body,
                                  headers={"X-Model-API-Key": fixture_key})
                self.assertEqual(saved.json(), {**body, "key_saved": True})
                protected = first.app.state.store.model_settings()["protected_key"]
                self.assertIsInstance(protected, bytes)
                self.assertNotIn(fixture_key.encode(), protected)

            with TestClient(create_app(path)) as reopened:
                settings = reopened.get("/api/model/settings")
                self.assertEqual(settings.json(), {**body, "key_saved": True})
                self.assertNotIn(fixture_key, settings.text)

                new_model = reopened.put("/api/model/settings", json={**body, "model": "qwen-max"})
                self.assertEqual(new_model.json(), {"provider": "dashscope", "model": "qwen-max",
                                                   "key_saved": True})
                with patch("deepvocal.model.httpx.post", return_value=response) as outbound:
                    chat = reopened.post("/api/chat", json={"query": "什么是换声区？", "intent": "qa",
                                                           "use_saved_model": True})
                    self.assertEqual(chat.json()["answer"], "已保存模型的回答 [register]")
                    self.assertEqual(outbound.call_args.kwargs["headers"]["Authorization"],
                                     f"Bearer {fixture_key}")
                    self.assertEqual(outbound.call_args.kwargs["json"]["model"], "qwen-max")
                    wrong_provider = reopened.post("/api/model/test",
                                                   json={**body, "provider": "openai", "use_saved_model": True})
                    self.assertEqual(wrong_provider.status_code, 409)
                    self.assertEqual(outbound.call_count, 1)
                cleared = reopened.put("/api/model/settings", json={"provider": "local"})
                self.assertEqual(cleared.json(), {"provider": "local", "model": "", "key_saved": False})

    def test_lyrics_provider(self):
        with patch.dict(os.environ, {"DEEPVOCAL_SINGING_URL": "http://example.invalid/sing"}):
            wav = self.client.get("/api/melodies/triad/guide.wav").content
            response = httpx.Response(200, content=wav, request=httpx.Request("POST", "http://example.invalid/sing"))
            with patch("deepvocal.audio.httpx.stream") as outbound:
                outbound.return_value.__enter__.return_value = response
                chat = self.client.post("/api/chat", json={
                    "query": "合成", "intent": "singing", "melody_id": "triad", "lyrics": "啦啦啦啦啦",
                }).json()
                job = self.wait_for(lambda: self.status(chat["job"]["id"]))
            self.assertEqual(job["result"]["type"], "singing")
            self.assertEqual(outbound.call_args.kwargs["json"]["midi_notes"], [60, 64, 67, 64, 60])
            self.assertEqual(self.client.get(job["result"]["audio_url"]).content, wav)
