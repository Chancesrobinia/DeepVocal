import os
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx

from deepvocal.main import create_app
from deepvocal.model import ModelError


class DeepVocalAPI(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
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

    def test_recording_plan_and_cross_session_memory(self):
        melody = self.client.get("/api/melodies/warmup/guide.wav?transpose=-1")
        recording = self.client.post("/api/audio", files={"file": ("guide.wav", melody.content, "audio/wav")})
        self.assertEqual(recording.status_code, 201)
        chat = self.client.post("/api/chat", json={
            "query": "测评并安排练习", "audio_id": recording.json()["id"], "days": 2, "minutes": 10,
        }).json()
        self.assertEqual(chat["steps"], ["assessment", "practice_plan"])
        job = self.wait_for(lambda: self.status(chat["job"]["id"]))
        self.assertEqual(job["state"], "succeeded", job)
        report = job["result"]
        self.assertEqual(report["matched_notes"], 8)
        self.assertAlmostEqual(report["median_pitch_error_cents"], -100, delta=10)
        self.assertEqual(report["plan"]["focus"], "pitch_flat")
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

    def test_knowledge_and_guide_generation(self):
        qa = self.client.post("/api/chat", json={"query": "什么是换声区？", "intent": "qa"}).json()
        self.assertEqual(qa["evidence"][0]["id"], "register")
        self.assertTrue(any(d["path"] for d in qa["evidence"]))
        fake = self.client.post("/api/audio", files={"file": ("bad.wav", b"bad", "audio/wav")})
        self.assertEqual(fake.status_code, 422)
        chat = self.client.post("/api/chat", json={"query": "生成示范音频", "intent": "singing"}).json()
        job = self.wait_for(lambda: self.status(chat["job"]["id"]))
        self.assertEqual(job["result"]["type"], "guide")
        self.assertTrue(self.client.get(job["result"]["audio_url"]).content.startswith(b"RIFF"))

    def test_knowledge_lookup_survives_chat_model_error(self):
        with patch("deepvocal.model.Model.chat", side_effect=ModelError("模型暂不可用。")):
            response = self.client.post("/api/chat", json={
                "query": "换声区是什么？", "provider": "dashscope", "model": "qwen-plus",
            }, headers={"X-Model-API-Key": "mock"})
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answer"]
        self.assertIn("模型暂不可用。", answer)
        self.assertIn("[register]", answer)

    def test_qa_shows_only_cited_evidence_and_withholds_unsubstantiated_answers(self):
        headers = {"X-Model-API-Key": "mock"}
        query = {"query": "什么是换声区？", "intent": "qa", "provider": "dashscope", "model": "qwen-plus"}
        for reply, expected_ids in [("换声区是一段音域。[register]", ["register"]),
                                    ("NO_EVIDENCE", []), ("无依据答案。[unknown]", [])]:
            with self.subTest(reply=reply), patch("deepvocal.model.Model.chat", return_value=reply):
                response = self.client.post("/api/chat", json=query, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual([item["id"] for item in result["evidence"]], expected_ids)
                if not expected_ids:
                    self.assertIn("没有找到能直接回答", result["answer"])

    def test_saved_sessions_and_paged_turns_with_job_results(self):
        first = self.client.post("/api/chat", json={"query": "什么是换声区？", "intent": "qa"}).json()
        self.client.post("/api/chat", json={"query": "另一个会话", "intent": "qa"})
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
        first = self.client.post("/api/chat", json={"query": "什么是换声区？", "intent": "qa"}).json()
        session = first["session_id"]
        second = self.client.post("/api/chat", json={
            "session_id": session, "query": "音分是什么？", "intent": "qa",
        }).json()
        other = self.client.post("/api/chat", json={"query": "什么是拖拍？", "intent": "qa"}).json()
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
        first = self.client.post("/api/chat", json={"query": "什么是换声区？", "intent": "qa"}).json()
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
