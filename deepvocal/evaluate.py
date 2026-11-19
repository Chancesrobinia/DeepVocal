import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

from .corpus import DATA_DIR
from .retrieval import build_retriever
from .storage import Storage


def main():
    parser = argparse.ArgumentParser(description="Compare evidence retrieval on labeled QA cases.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8-sig") as file:
        items = [(line_no, json.loads(raw)) for line_no, raw in enumerate(file, 1) if raw.strip()]
    if not items or (args.expected_count is not None and len(items) != args.expected_count):
        parser.error(f"Unexpected number of examples: {len(items)}")
    store = Storage(os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR)
    retriever = build_retriever(store, index_filename="eval-knowledge.db")
    for line, item in items:
        ids = item.get("relevant_chunk_ids", [])
        if not item.get("question") or not ids or set(ids) - retriever.rows.keys():
            parser.error(f"Line {line}: missing question or invalid relevant_chunk_ids")
    results = {}
    try:
        for mode in ("vector", "rerank", "graph"):
            recall = mrr = ndcg = 0.0
            started = perf_counter()
            misses = []
            for _, item in items:
                relevant = set(item["relevant_chunk_ids"])
                retrieved = retriever.search(item["question"], mode)
                ranks = [rank for rank, row in enumerate(retrieved, 1) if row["id"] in relevant]
                recall += len(ranks) / len(relevant)
                mrr += 1 / ranks[0] if ranks else 0
                ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(3, len(relevant)) + 1))
                ndcg += sum(1 / math.log2(rank + 1) for rank in ranks) / ideal
                if not ranks:
                    misses.append(item["question"])
            count = len(items)
            results[mode] = {"recall@3": round(recall / count, 4), "mrr@3": round(mrr / count, 4),
                             "ndcg@3": round(ndcg / count, 4), "mean_ms": round(1000 * (perf_counter() - started) / count, 2),
                             "misses": misses}
    finally:
        if retriever.client:
            retriever.client.close()
    print(json.dumps({"backend": retriever.mode, "embedding_model": retriever.embedding_id,
                      "count": len(items), "results": results}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
