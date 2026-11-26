import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from time import perf_counter

from .corpus import DATA_DIR
from .knowledge import CROSS_MODES
from .retrieval import build_retriever
from .storage import Storage


MODES = ("vector", "rerank", "graph")
TOP_K = 3
METRICS = ("recall@3", "mrr@3", "ndcg@3", "complete@3", "mean_ms")


def measure(ranks, relevant_count, elapsed_ms):
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(TOP_K, relevant_count) + 1))
    return {
        "recall@3": len(ranks) / relevant_count,
        "mrr@3": 1 / ranks[0] if ranks else 0,
        "ndcg@3": sum(1 / math.log2(rank + 1) for rank in ranks) / ideal,
        "complete@3": int(len(ranks) == relevant_count),
        "mean_ms": elapsed_ms,
    }


def summarize(cases, modes):
    return {
        mode: {
            **{key: round(sum(case["modes"][mode][key] for case in cases) / len(cases), 4)
               for key in METRICS},
            "misses": [case["question"] for case in cases if not case["modes"][mode]["ranks"]],
        }
        for mode in modes
    }


def main():
    parser = argparse.ArgumentParser(description="Compare evidence retrieval on labeled QA cases.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--modes", choices=(*MODES, *CROSS_MODES), nargs="+", default=MODES)
    parser.add_argument("--reuse-query-embeddings", action="store_true",
                        help="Reuse the same vectors across modes. Reported retrieval time excludes embedding.")
    parser.add_argument("--report", type=Path, help="Write a UTF-8 JSON report with the per-question rankings.")
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8-sig") as file:
        items = [(line_no, json.loads(raw)) for line_no, raw in enumerate(file, 1) if raw.strip()]
    if not items or (args.expected_count is not None and len(items) != args.expected_count):
        parser.error(f"Unexpected number of examples: {len(items)}")
    store = Storage(os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR)
    retriever = build_retriever(store, index_filename="eval-knowledge.db")
    modes = tuple(dict.fromkeys(args.modes))
    cases = []
    embedding_timing = None
    try:
        if set(modes).intersection(CROSS_MODES) and not retriever.reranker:
            parser.error("Configure DEEPVOCAL_RERANK_MODEL and a DashScope API key for Cross-Encoder evaluation.")
        for line, item in items:
            ids = item.get("relevant_chunk_ids", [])
            if not item.get("question") or not ids or set(ids) - retriever.rows.keys():
                parser.error(f"Line {line}: missing question or invalid relevant_chunk_ids")
        if args.reuse_query_embeddings and retriever.embed:
            questions = list(dict.fromkeys(item["question"] for _, item in items))
            started = perf_counter()
            embeddings = dict(zip(questions, retriever.ordered_embeddings(questions).tolist()))
            embedding_timing = {"queries": len(questions), "batch_ms": round(1000 * (perf_counter() - started), 2)}
            retriever.embed = lambda texts: [embeddings[text] for text in texts]

        for line, item in items:
            case = {"id": item.get("id", line), "group": item.get("group", "ungrouped"),
                    "question": item["question"], "relevant_chunk_ids": item["relevant_chunk_ids"], "modes": {}}
            relevant = set(item["relevant_chunk_ids"])
            for mode in modes:
                started = perf_counter()
                rows = retriever.search(case["question"], mode, top_k=TOP_K)
                elapsed_ms = 1000 * (perf_counter() - started)
                ranks = [rank for rank, row in enumerate(rows, 1) if row["id"] in relevant]
                case["modes"][mode] = {**measure(ranks, len(relevant), elapsed_ms), "ranks": ranks,
                                       "retrieved": [{"id": row["id"], "score": row["score"],
                                                      "graph_hops": len(row["path"])} for row in rows]}
                if mode in CROSS_MODES:
                    case["modes"][mode]["reranker"] = rows[0]["reranker"] if rows else "no_candidates"
            cases.append(case)
    finally:
        if retriever.client:
            retriever.client.close()
    summary = {"backend": retriever.mode, "embedding_model": retriever.embedding_id,
               "knowledge_chunks": len(retriever.rows), "count": len(cases), "results": summarize(cases, modes)}
    if cross_modes := set(modes).intersection(CROSS_MODES):
        summary["cross_encoder_usage"] = {
            mode: dict(Counter(case["modes"][mode]["reranker"] for case in cases))
            for mode in sorted(cross_modes)
        }
    if embedding_timing is not None:
        summary["shared_query_embeddings"] = embedding_timing
    if any(case["group"] != "ungrouped" for case in cases):
        by_group = {}
        for case in cases:
            by_group.setdefault(case["group"], []).append(case)
        summary["groups"] = {name: {"count": len(cases_in_group), "results": summarize(cases_in_group, modes)}
                             for name, cases_in_group in by_group.items()}
    if {"rerank", "graph"} <= set(modes):
        changes = Counter()
        for case in cases:
            before = case["modes"]["rerank"]["ndcg@3"]
            after = case["modes"]["graph"]["ndcg@3"]
            changes["improved" if after > before else "worse" if after < before else "tied"] += 1
        summary["graph_vs_rerank_ndcg@3"] = {key: changes[key] for key in ("improved", "worse", "tied")}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({**summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
