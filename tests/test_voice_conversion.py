import json
import os
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import numpy as np

from deepvocal import audio
from deepvocal.main import create_app
from deepvocal.voice_conversion import convert_song


class VoiceConversionAPI(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DEEPVOCAL_SEED_VC_TOKEN": "fixture-token"}, clear=True))
        folder = self.enterContext(TemporaryDirectory())
        self.client = self.enterContext(TestClient(create_app(folder)))
        self.source = audio.guide([60, 64, 67], 120, 0)
        self.timbre = audio.guide([57] * 40, 90, 0)
        samples, rate = audio.read_wave(self.source)
        self.output = audio.to_wave(np.repeat(samples, 2), rate * 2)

    def upload(self, wav):
        response = self.client.post("/api/audio", files={"file": ("synthetic.wav", wav, "audio/wav")})
        response.raise_for_status()
        return response.json()["id"]

    def upload_reference(self, wav, title, filename="reference.wav"):
        response = self.client.post("/api/audio/reference", files={"file": (filename, wav, "audio/wav")},
                                    data={"song_title": title})
        response.raise_for_status()
        return response.json()["id"]

    def wait_job(self, key):
        deadline = monotonic() + 5
        while monotonic() < deadline:
            result = self.client.get(f"/api/jobs/{key}").json()
            if result["state"] in ("succeeded", "failed"):
                return result
            sleep(0.02)
        self.fail("Voice conversion did not finish")

    def call(self, handler, *, source=None):
        payload = {"source_vocal_audio_id": self.upload(self.source if source is None else source),
                   "timbre_audio_id": self.upload(self.timbre), "semitone_shift": -2}
        transport = httpx.MockTransport(handler)
        client_class = httpx.Client
        with patch("deepvocal.voice_conversion.httpx.Client",
                   side_effect=lambda **options: client_class(transport=transport, **options)):
            response = self.client.post("/api/voice-conversions", json=payload)
            self.assertEqual(response.status_code, 202, response.text)
            return self.wait_job(response.json()["job"]["id"])

    def upstream(self, *, failure=None, output_url=None):
        calls = []
        current_api = ("source_audio_path", "target_audio_path", "diffusion_steps", "length_adjust",
                       "inference_cfg_rate", "f0_condition", "auto_f0_adjust", "pitch_shift")
        api_path = "/gradio_api/"
        file_path = api_path + "file=/tmp/gradio/complete.wav"
        source_samples, source_rate = audio.read_wave(self.source)
        timbre_samples, timbre_rate = audio.read_wave(self.timbre)
        expected_source = audio.to_wave(source_samples, source_rate)
        expected_timbre = audio.to_wave(timbre_samples[:25 * timbre_rate], timbre_rate)

        def respond(request):
            self.assertEqual(request.url.host, "plachta-seed-vc.hf.space")
            self.assertEqual(request.headers.get("Authorization"), "Bearer fixture-token")
            path = request.url.path
            calls.append(path)
            if path == "/config":
                return httpx.Response(200, json={"api_prefix": api_path.rstrip("/"), "dependencies": [
                    {"api_name": "predict", "id": 3, "backend_fn": True},
                    {"api_name": "predict_1", "id": 12, "backend_fn": True},
                ]})
            if path == api_path + "info":
                return httpx.Response(200, json={"named_endpoints": {
                    "/predict": {"parameters": [{"parameter_name": "convert_style"}], "returns": []},
                    "/predict_1": {"parameters": [{"parameter_name": key} for key in current_api],
                                   "returns": [{"label": "Stream Output Audio"}, {"label": "Full Output Audio"}]},
                }})
            if path == api_path + "upload":
                self.assertIn(b'filename="source-vocal.wav"', request.content)
                self.assertIn(expected_source, request.content)
                self.assertLess(request.content.index(expected_source), request.content.index(expected_timbre))
                return httpx.Response(200, json=["/tmp/gradio/source.wav", "/tmp/gradio/timbre.wav"])
            if path == api_path + "queue/join":
                body = json.loads(request.content)
                self.assertEqual(body["fn_index"], 12)
                self.assertEqual(body["data"], [
                    {"path": "/tmp/gradio/source.wav", "meta": {"_type": "gradio.FileData"}},
                    {"path": "/tmp/gradio/timbre.wav", "meta": {"_type": "gradio.FileData"}},
                    30, 1.0, 0.7, True, False, -2,
                ])
                return httpx.Response(200, json={"event_id": "cover-event"})
            if path == api_path + "queue/data":
                self.assertTrue(request.url.params.get("session_hash"))
                data = {"error": failure} if failure else {"data": [None, {
                    "url": output_url or f"https://{request.url.host}{file_path}", "is_stream": False}]}
                event = {"msg": "process_completed", "event_id": "cover-event", "success": not failure, "output": data}
                return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")
            if path == file_path:
                return httpx.Response(200, content=self.output)
            self.fail(f"Unexpected Seed-VC URL {request.url}")

        return respond, calls

    def test_conversion_preserves_roles_and_stores_complete_wav(self):
        handler, calls = self.upstream()
        result = self.call(handler)
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual(result["kind"], "voice_conversion")
        self.assertEqual(result["result"], {"type": "cover", "provider": "seed-vc",
            "audio_url": f"/api/audio/{result['id']}", "duration_seconds": 1.5,
            "sample_rate_hz": 44100, "semitone_shift": -2})
        media = self.client.get(result["result"]["audio_url"])
        self.assertEqual(media.content, self.output)
        self.assertTrue(media.headers["content-type"].startswith("audio/wav"))
        self.assertEqual(calls[-1], "/gradio_api/file=/tmp/gradio/complete.wav")

    def test_upstream_quota_is_reported_and_no_audio_is_published(self):
        handler, _ = self.upstream(failure="You have exceeded your ZeroGPU runs limit.")
        result = self.call(handler)
        self.assertEqual(result["state"], "failed", result)
        self.assertIn("GPU 配额", result["error"])
        self.assertEqual(self.client.get(f"/api/audio/{result['id']}").status_code, 404)

    def test_output_url_must_be_served_by_the_configured_upstream(self):
        handler, calls = self.upstream(output_url="https://other.example/gradio_api/file=/tmp/cover.wav")
        result = self.call(handler)
        self.assertEqual(result["state"], "failed", result)
        self.assertIn("音频地址无效", result["error"])
        self.assertEqual(calls[-1], "/gradio_api/queue/data")

    def test_silent_audio_is_rejected_before_the_remote_call(self):
        handler, calls = self.upstream()
        result = self.call(handler, source=audio.to_wave(np.zeros(8000), 8000))
        self.assertEqual(result["state"], "failed", result)
        self.assertIn("可听见", result["error"])
        self.assertEqual(calls, [])

    def test_chat_cover_reuses_recording_when_user_supplies_the_missing_title(self):
        self.upload_reference(self.source, "山谷回声")
        initial = self.client.post("/api/chat", json={"query": "用我的音色翻唱", "audio_id": self.upload(self.timbre)}).json()
        self.assertEqual(initial["steps"], ["singing"])
        self.assertEqual(initial["singing_agent"]["issue"], "song_title_required")
        self.assertNotIn("job", initial)

        with patch("deepvocal.voice_conversion.convert_singing_voice", return_value=self.output) as convert:
            result = self.client.post("/api/chat", json={"query": "山谷回声", "session_id": initial["session_id"]}).json()
            self.assertEqual(result["singing_agent"]["tools"], ["search_song_reference", "convert_singing_voice"])
            job = self.wait_job(result["job"]["id"])
        self.assertEqual(job["state"], "succeeded", job)
        self.assertEqual(job["result"]["song_title"], "山谷回声")
        self.assertTrue(convert.call_args.kwargs["local_only"])
        self.assertEqual(convert.call_args.args[1], self.timbre)
        self.assertEqual(self.client.get(job["result"]["audio_url"]).content, self.output)

    def test_chat_uses_reference_attachment_without_searching_duplicate_versions(self):
        title = "山谷回声"
        self.upload_reference(self.source, title)
        alternate = audio.guide([57, 62, 69], 120, 0)
        alternate_id = self.upload_reference(alternate, title, "alternate.wav")
        payload = {"query": f"用我上传的录音唱《{title}》", "audio_id": self.upload(self.timbre)}
        ambiguous = self.client.post("/api/chat", json=payload).json()
        self.assertEqual(ambiguous["singing_agent"]["issue"], "reference_ambiguous")
        self.assertNotIn("job", ambiguous)

        with patch("deepvocal.voice_conversion.convert_singing_voice", return_value=self.output) as convert:
            attached = self.client.post("/api/chat", json={**payload, "reference_audio_id": alternate_id}).json()
            self.assertEqual(attached["singing_agent"]["tools"], ["convert_singing_voice"])
            self.assertEqual(self.wait_job(attached["job"]["id"])["state"], "succeeded")
        samples, rate = audio.read_wave(alternate)
        self.assertEqual(convert.call_args.args[0], audio.to_wave(samples, rate))

    def test_chat_cover_uses_grounded_model_tool_arguments(self):
        self.upload_reference(self.source, "山谷回声")
        calls = []

        def tool_reply(messages, *, tools, tool_choice):
            self.assertEqual(tool_choice, "auto")
            name = tools[0]["function"]["name"]
            calls.append(name)
            args = {"song_title": "山谷回声"} if name == "search_song_reference" else {}
            if not args:
                self.assertEqual(json.loads(messages[-1]["content"])["status"], "reference_found")
            return {"tool_calls": [{"id": name, "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}]}

        with (patch("deepvocal.model.Model.chat_message", side_effect=tool_reply),
              patch("deepvocal.voice_conversion.convert_singing_voice", return_value=self.output)):
            result = self.client.post("/api/chat", headers={"X-Model-API-Key": "mock"}, json={
                "query": "山谷回声这首歌，请用我上传的录音作为翻唱音色。", "intent": "singing",
                "audio_id": self.upload(self.timbre), "provider": "dashscope", "model": "qwen-plus",
            }).json()
            self.assertEqual(result["singing_agent"]["conversion_decision"], "model")
            self.assertEqual(self.wait_job(result["job"]["id"])["state"], "succeeded")
        self.assertEqual(calls, ["search_song_reference", "convert_singing_voice"])

    def test_long_song_stitches_windows_without_changing_duration(self):
        rate = 22050
        source = audio.to_wave(np.full(rate * 63, 0.3, dtype=np.float32), rate)
        progress = []
        with patch("deepvocal.voice_conversion.convert_singing_voice", side_effect=lambda wav, *_args, **_kw: wav) as convert:
            output = convert_song(source, self.timbre, semitone_shift=0,
                                  progress=lambda done, total: progress.append((done, total)))
        samples, output_rate = audio.read_wave(output, 64)
        self.assertEqual(convert.call_count, 4)
        self.assertEqual(len(samples), 63 * output_rate)
        np.testing.assert_allclose(samples, 0.3, atol=0.001)
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])

    def test_distinct_uploads_and_shift_are_required(self):
        key = self.upload(self.source)
        self.assertEqual(self.client.post("/api/voice-conversions", json={
            "source_vocal_audio_id": key, "timbre_audio_id": key}).status_code, 422)
        self.assertEqual(self.client.post("/api/voice-conversions", json={
            "source_vocal_audio_id": key, "timbre_audio_id": "missing"}).status_code, 404)
        self.assertEqual(self.client.post("/api/voice-conversions", json={
            "source_vocal_audio_id": key, "timbre_audio_id": "missing", "semitone_shift": 13}).status_code, 422)
