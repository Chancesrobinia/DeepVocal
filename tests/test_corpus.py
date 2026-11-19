from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deepvocal.ingest import main as ingest
from deepvocal.main import create_app
from deepvocal.storage import Storage


class CorpusTest(unittest.TestCase):
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

            with TestClient(create_app(root)) as client, patch("deepvocal.model.Model.chat") as chat:
                self.assertEqual(client.get("/api/health").json()["imported_documents"], 1)
                response = client.post("/api/chat", json={
                    "query": "蓝桉热身", "intent": "qa", "provider": "dashscope", "model": "qwen-plus",
                }, headers={"X-Model-API-Key": "mock"})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["evidence"][0]["source"], "练声笔记 · 蓝桉热身")
                self.assertIn(text, result["answer"])
                chat.assert_not_called()
                self.assertEqual(Storage(root).history(result["session_id"])[0]["answer"], "")
                turns = client.get(f"/api/sessions/{result['session_id']}/turns").json()["turns"]
                self.assertIn(text, turns[0]["result"]["answer"])
