"""Collect paired answers from the saved chat model and the current QA workflow."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from .corpus import DATA_DIR
from .main import ChatInput, requested_model
from .model import ModelError
from .retrieval import build_retriever
from .storage import Storage
from .workflow import build_workflow


DIRECT_SYSTEM = "你是声乐和乐理助教。用中文直接回答用户的问题。如果无法确认某项信息，请说明。"


def collect_pair(case, model, retriever, first, retrieval):
    request = ChatInput(query=case["question"], intent="qa", retrieval=retrieval, use_saved_model=True).model_dump()
    candidates = []
    trace = {}

    def search(*args):
        rows = retriever.search(*args)
        candidates.extend({k: row[k] for k in ("id", "title", "text", "score", "allow_cloud", "reranker") if k in row}
                          for row in rows)
        return rows

    def decide(messages, **options):
        try:
            message = model.chat_message(messages, **options)
        except ModelError as exc:
            trace["model_error"] = str(exc)
            raise
        if not message.get("tool_calls"):
            trace["raw_model_answer"] = message.get("content")
        return message

    workflow = build_workflow(None, SimpleNamespace(search=search, graph=retriever.graph,
                                                    local_only_count=retriever.local_only_count), None)
    outputs = {}
    for mode in (first, "rag" if first == "direct" else "direct"):
        started = perf_counter()
        if mode == "direct":
            try:
                output = {"answer": model.chat([
                    {"role": "system", "content": DIRECT_SYSTEM},
                    {"role": "user", "content": case["question"]},
                ])}
            except ModelError as exc:
                output = {"model_error": str(exc)}
        else:
            result = workflow.invoke({
                "request": request, "history": [],
                "model": SimpleNamespace(chat_message=decide, provider=model.provider, name=model.name),
            })
            output = {"answer": result["answer"], "evidence_ids": [row["id"] for row in result["evidence"]],
                      "candidates": candidates, "retrieval": result["rag"], **trace}
        outputs[mode] = {**output, "elapsed_ms": round((perf_counter() - started) * 1000)}
    return {**case, "first": first, **outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--retrieval", choices=("graph", "cross_rerank", "cross_graph"), default="graph")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("The dataset needs at least one case and distinct case IDs.")
    store = Storage(os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR)
    model = requested_model(store, ChatInput(query="eval", use_saved_model=True), None)
    if model is None:
        parser.error("Save a chat model in DeepVocal before running the evaluation.")
    retriever = build_retriever(store, index_filename="eval-knowledge.db")
    try:
        for case in cases:
            if set(case["evidence_ids"]) - retriever.rows.keys():
                parser.error(f"Evidence passages are missing for {case['id']}.")
        if args.retrieval != "graph" and not retriever.reranker:
            parser.error("Configure DEEPVOCAL_RERANK_MODEL and a DashScope API key for Cross-Encoder evaluation.")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(collect_pair, case, model, retriever, "rag" if i % 2 else "direct", args.retrieval)
                       for i, case in enumerate(cases)]
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
                print(f"{result['id']}  direct={result['direct']['elapsed_ms']}ms  rag={result['rag']['elapsed_ms']}ms")
    finally:
        if retriever.client:
            retriever.client.close()
    report = {
        "model": model.name, "provider": model.provider, "embedding_model": retriever.embedding_id,
        "backend": retriever.mode, "retrieval": args.retrieval, "knowledge_chunks": len(retriever.rows),
        "direct_system": DIRECT_SYSTEM, "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
