import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from deepvocal.knowledge import Retriever
from deepvocal.model import Model


class RerankTest(unittest.TestCase):
    def setUp(self):
        self.model = Model("dashscope", "qwen3.7-text-rerank", "test-key")

    def test_maps_provider_indices_to_original_graph_evidence(self):
        retriever = Retriever(reranker=self.model)

        def response(url, *, json, **kwargs):
            self.assertTrue(url.endswith("/services/rerank/text-rerank/text-rerank"))
            self.assertEqual(json["input"]["query"], "换声区怎么练？")
            self.assertEqual(json["parameters"]["top_n"], 1)
            index = next(i for i, text in enumerate(json["input"]["documents"]) if text.startswith("唇颤音练习\n"))
            return httpx.Response(200, json={"output": {"results": [{"index": index, "relevance_score": 0.9}]}})

        with patch("deepvocal.model.httpx.post", side_effect=response):
            rows = retriever.search("换声区怎么练？", "cross_graph", top_k=1)
        self.assertEqual(rows[0]["id"], "lip")
        self.assertEqual(len(rows[0]["path"]), 2)
        self.assertEqual(rows[0]["score"], 0.9)
        self.assertEqual(rows[0]["reranker"], self.model.name)

    def test_invalid_response_falls_back_to_rules(self):
        retriever = Retriever(reranker=self.model)
        invalid = {"output": {"results": [{"index": -1, "relevance_score": 0.9}]}}
        with patch("deepvocal.model.httpx.post", return_value=httpx.Response(200, json=invalid)):
            with self.assertLogs("deepvocal.knowledge", "WARNING"):
                rows = retriever.search("起音怎么练？", "cross_rerank")
        baseline = retriever.search("起音怎么练？", "rerank")
        self.assertEqual(rows, [{**row, "reranker": "rules"} for row in baseline])

    def test_local_candidate_keeps_entire_batch_on_device(self):
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            (corpus / "private.json").write_text(json.dumps({
                "id": "private", "title": "练习笔记", "use": "local",
                "chunks": [{"id": "private-note", "text": "琥珀是练习暗号。"}],
            }), encoding="utf-8")
            retriever = Retriever(corpus_dir=corpus, reranker=self.model)
            with patch("deepvocal.model.httpx.post") as cloud:
                rows = retriever.search("琥珀练习是什么？", "cross_rerank")
            cloud.assert_not_called()
            self.assertEqual(rows[0]["id"], "private-note")
            self.assertEqual(rows[0]["reranker"], "rules")
