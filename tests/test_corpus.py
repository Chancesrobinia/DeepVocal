from base64 import b64decode
from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from deepvocal.corpus import Chunk, Document, Figure
from deepvocal.ingest import main as ingest, write_document
from deepvocal.main import create_app
from deepvocal.storage import Storage


MODEL = {"provider": "dashscope", "model": "qwen-plus"}
MODEL_HEADERS = {"X-Model-API-Key": "mock"}


def tool_reply(query):
    return {"role": "assistant", "tool_calls": [{"id": "call_search", "type": "function",
            "function": {"name": "search_knowledge", "arguments": json.dumps({"query": query})}}]}


class CorpusTest(unittest.TestCase):
    def test_local_figure_retrieval_and_asset_access(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPVOCAL_ACCESS_TOKEN": "test-access"}, clear=True):
            root = Path(directory)
            chunk = Chunk(id="chart-text", text="An illustration of the larynx.", page=12, page_label="6",
                          page_end=13, page_end_label="7", summary="紫杉发声示意图。", entities=["紫杉声带投影"],
                          figures=[Figure(id="chart", caption="解剖图", description="紫杉声带投影")])
            write_document(root / "knowledge" / "book.json", Document(id="book", title="声乐资料", chunks=[chunk]))
            assets = root / "knowledge-assets" / "book"
            assets.mkdir(parents=True)
            png = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWMsAAAAASUVORK5CYII=")
            (assets / "chart.png").write_bytes(png)
            (assets / "undeclared.png").write_bytes(png)
            headers = {"X-DeepVocal-Token": "test-access", **MODEL_HEADERS}
            with (TestClient(create_app(root)) as client,
                  patch("deepvocal.model.Model.chat_message", side_effect=[
                      tool_reply("紫杉声带投影"), {"content": "可在本机查看相关图版。"},
                  ])):
                response = client.post("/api/chat", json={"query": "紫杉声带投影", "intent": "qa", **MODEL}, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["evidence"], [])
                figure = result["related_figures"][0]
                self.assertEqual(figure["id"], chunk.id)
                self.assertNotIn(chunk.summary, result["answer"])
                self.assertIn("PDF 第 12–13 页 · 页标 6–7", figure["source"])
                url = figure["figures"][0]["url"]
                self.assertEqual(client.get(url).status_code, 401)
                image = client.get(url, headers=headers)
                self.assertEqual(image.headers["content-type"], "image/png")
                self.assertEqual(image.content, png)
                self.assertEqual(client.get(url.replace("/chart", "/undeclared", 1), headers=headers).status_code, 404)
                self.assertEqual(client.get(url.rsplit("/", 1)[0] + "/undeclared", headers=headers).status_code, 404)

    @unittest.skipUnless(find_spec("pymilvus"), "requires the optional Milvus dependencies")
    def test_chat_recovers_from_milvus_disconnect(self):
        from pymilvus.exceptions import MilvusException

        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with TestClient(create_app(Path(directory))) as client:
                retriever = client.app.state.retriever
                retriever.client = Mock()
                retriever.collection = "test"
                retriever.client.search.side_effect = MilvusException(message="connection unavailable")
                with (self.assertLogs("deepvocal.knowledge", "WARNING"),
                      patch("deepvocal.model.Model.chat_message", side_effect=[tool_reply("什么是换声区？"),
                                                                             {"content": "示例回答 [register]"}])):
                    response = client.post("/api/chat", json={
                        "query": "什么是换声区？", "intent": "qa", "retrieval": "graph", **MODEL,
                    }, headers=MODEL_HEADERS)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["evidence"][0]["id"], "register")
                self.assertEqual(retriever.client.search.call_args.kwargs["timeout"], 3)

    def test_imported_local_passage_is_retrievable_and_excluded_from_cloud_context(self):
        text = "蓝桉热身是自编练习，暗号是棉云。"
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            source = root / "notes.md"
            source.write_text("# 蓝桉热身\n\n" + text, encoding="utf-8")
            args = ["ingest", str(source), "--id", "notes", "--title", "练声笔记",
                    "--output-dir", str(root / "knowledge")]
            with patch("sys.argv", args), redirect_stdout(StringIO()):
                ingest()

            with (TestClient(create_app(root)) as client,
                  patch("deepvocal.model.Model.chat_message", side_effect=[
                      tool_reply("蓝桉热身"), {"content": "可以先从轻松哼鸣开始热身。"},
                  ]) as chat):
                self.assertEqual(client.get("/api/health").json()["imported_documents"], 1)
                response = client.post("/api/chat", json={
                    "query": "蓝桉热身", "intent": "qa", **MODEL,
                }, headers=MODEL_HEADERS)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["evidence"], [])
                self.assertEqual(result["qa_generation"], {"mode": "model_fallback", **MODEL})
                self.assertIn("轻松哼鸣", result["answer"])
                self.assertNotIn(text, result["answer"])
                self.assertEqual([call.kwargs.get("tool_choice") for call in chat.call_args_list], ["auto", None])
                for call in chat.call_args_list:
                    self.assertNotIn(text, json.dumps(call.args[0], ensure_ascii=False))
                self.assertTrue(result["rag"]["used"])
                turns = client.get(f"/api/sessions/{result['session_id']}/turns").json()["turns"]
                self.assertEqual(result["answer"], turns[0]["result"]["answer"])

    def test_mixed_followup_sends_only_public_evidence_and_keeps_generation_mode(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            private = Chunk(id="private-falsetto", section="假声训练与假声练习", entities=["假声"],
                            text="假声训练的本地笔记，测试暗号是紫杉石。",
                            figures=[Figure(id="private-chart", caption="本地图版", description="假声训练的紫杉石图版")])
            write_document(root / "knowledge" / "private.json",
                           Document(id="private", title="私人笔记", chunks=[private]))
            with TestClient(create_app(root)) as client:
                session = "legacy-private-turn"
                client.app.state.store.record_turn(session, "假声训练是什么？", {
                    "steps": ["qa"], "answer": private.text, "evidence": [{"id": private.id, "allow_cloud": False}],
                    "qa_generation": {"mode": "evidence_only"},
                })
                body = {"query": "应该怎么训练？", "intent": "qa", "session_id": session,
                        **MODEL}

                def generate(messages, *, tool_choice, **kwargs):
                    prompt = json.loads(messages[1]["content"])
                    self.assertEqual(prompt["question"], body["query"])
                    self.assertEqual(prompt["history"][-1]["answer"], "")
                    self.assertNotIn("紫杉石", messages[-1]["content"])
                    self.assertNotIn(private.id, messages[-1]["content"])
                    self.assertNotIn("private-chart", messages[-1]["content"])
                    if tool_choice == "auto":
                        return tool_reply("假声应该怎么训练？")
                    evidence = json.loads(messages[-1]["content"])["evidence"]
                    self.assertIn("falsetto-practice", {row["id"] for row in evidence})
                    return {"content": "先用呜轻声下滑，再尝试短下行琶音。[ falsetto-practice ]"}

                with patch("deepvocal.model.Model.chat_message", side_effect=generate) as chat:
                    response = client.post("/api/chat", json=body, headers=MODEL_HEADERS)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(chat.call_count, 2)
                result = response.json()
                expected = {"mode": "model", "model": "qwen-plus", "provider": "dashscope"}
                self.assertEqual(result["qa_generation"], expected)
                self.assertEqual(result["rag"]["query"], "假声应该怎么训练？")
                self.assertEqual([row["id"] for row in result["evidence"]], ["falsetto-practice"])
                self.assertIn("[falsetto-practice]", result["answer"])
                self.assertEqual([row["id"] for row in result["related_figures"]], [private.id])
                self.assertNotIn("text", result["related_figures"][0])
                turns = client.get(f"/api/sessions/{session}/turns").json()["turns"]
                self.assertEqual(turns[-1]["result"]["qa_generation"], expected)
                self.assertEqual(turns[-1]["result"]["related_figures"], result["related_figures"])
                self.assertIn("下行琶音", Storage(root).history(session)[-1]["answer"])

                with patch("deepvocal.model.Model.chat_message", side_effect=[
                    tool_reply("假声训练是什么？"), {"content": "NO_EVIDENCE"}, {"content": "可先做轻声下滑。"},
                ]) as chat:
                    retry = client.post("/api/chat", json={**body, "query": "假声训练是什么？"}, headers=MODEL_HEADERS).json()
                self.assertEqual(retry["qa_generation"], {"mode": "model_fallback", **MODEL})
                self.assertEqual(retry["evidence"], [])
                self.assertEqual([row["id"] for row in retry["related_figures"]], [private.id])
                self.assertNotIn(private.id, json.dumps(chat.call_args.args[0], ensure_ascii=False))
