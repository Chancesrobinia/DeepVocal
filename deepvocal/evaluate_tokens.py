"""Measure provider-reported token usage for the four QA retrieval modes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from unittest.mock import patch

from .corpus import DATA_DIR
from .knowledge import CROSS_MODES
from .main import ChatInput, requested_model
from .model import Model, ModelError
from .retrieval import build_retriever
from .storage import Storage
from .workflow import build_workflow


MODES = ("rerank", "graph", *CROSS_MODES)
STAGES = {"embeddings": "embedding", "text-rerank": "rerank", "completions": "generation"}
CHAT_STAGES = ("decision", "generation", "fallback")
FIELDS = ("input_tokens", "output_tokens", "total_tokens")
active_trace = ContextVar("token_evaluation_trace", default=None)


def summarize(cases):
    summary = {}
    for mode in MODES:
        rows = [case for case in cases if case["mode"] == mode and case["valid"]]
        stages = {}
        for stage in ("decision", *STAGES.values(), "fallback"):
            calls = [call for case in rows for call in case["calls"] if call["stage"] == stage]
            stages[stage] = {
                "calls": len(calls),
                **{field: sum(call[field] for call in calls) for field in FIELDS},
            }
        totals = {field: sum(stage[field] for stage in stages.values()) for field in FIELDS}
        summary[mode] = {
            "valid_cases": len(rows),
            "stages": stages,
            "total": totals,
            "mean_tokens_per_question": {
                field: round(totals[field] / len(rows), 2) for field in FIELDS
            } if rows else {},
            "mean_rerank_candidates": round(mean(case["rerank_candidates"] for case in rows), 2) if rows else None,
            "no_evidence_answers": sum(case["no_evidence"] for case in rows),
            "retrieval_used": sum(case["retrieval_used"] for case in rows),
        }
    return summary


def collect(case, mode, workflow, model):
    request = ChatInput(query=case["question"], intent="qa", retrieval=mode, use_saved_model=True)
    retrieval_stages = ["embedding", "rerank"] if mode in CROSS_MODES else ["embedding"]
    retries = []
    for _ in range(2):
        trace = {"calls": [], "retrieved": [], "rerank_candidates": 0}
        token = active_trace.set(trace)
        try:
            result = workflow.invoke({"request": request.model_dump(), "history": [], "model": model})
        finally:
            active_trace.reset(token)
        used = result["rag"]["used"]
        generation = result["qa_generation"]["mode"]
        fallback = generation == "model_fallback"
        base = ["decision", *retrieval_stages] if used else ["decision"]
        expected = [base + ["generation"] if used else base] if generation == "model" else []
        if used and fallback:
            expected = [base + ["fallback"], base + ["generation", "fallback"]]
        valid = ([call["stage"] for call in trace["calls"]] in expected
                 and all("total_tokens" in call for call in trace["calls"]))
        if used and mode in CROSS_MODES:
            valid = (valid and bool(trace["retrieved"])
                     and all(row.get("reranker") not in (None, "rules") for row in trace["retrieved"]))
        row = {"id": case["id"], "group": case["group"], "question": case["question"], "mode": mode, "valid": valid,
               **trace, "retrieval_used": used, "cited_ids": [item["id"] for item in result["evidence"]],
               "no_evidence": fallback, "prior_attempts": list(retries)}
        if valid:
            return row
        retries.append(trace["calls"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not items or (args.expected_count is not None and len(items) != args.expected_count):
        parser.error(f"Unexpected number of questions: {len(items)}")
    if args.workers < 1:
        parser.error("Use at least one worker.")

    store = Storage(os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR)
    model = requested_model(store, ChatInput(query="token evaluation", use_saved_model=True), None)
    if model is None or model.provider != "dashscope":
        parser.error("Save a DashScope chat model before measuring provider-reported token usage.")
    retriever = build_retriever(store, index_filename="eval-knowledge.db")
    original_call = Model.call

    def metered_call(client, path, payload, **kwargs):
        trace = active_trace.get()
        if trace is None:
            return original_call(client, path, payload, **kwargs)
        stage = STAGES[path.rsplit("/", 1)[-1]]
        if stage == "generation":
            stage = {"auto": "decision", "none": "generation"}.get(payload.get("tool_choice"), "fallback")
        entry = {"stage": stage, "model": client.name}
        trace["calls"].append(entry)
        if stage == "rerank":
            trace["rerank_candidates"] = len(payload["input"]["documents"])
        try:
            response = original_call(client, path, payload, **kwargs)
        except ModelError as exc:
            entry["error"] = str(exc)
            raise
        usage = response.get("usage") if isinstance(response, dict) else None
        entry["provider_usage"] = usage
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            incoming = usage.get("prompt_tokens", total if stage not in CHAT_STAGES else None)
            outgoing = usage.get("completion_tokens", 0 if stage not in CHAT_STAGES else None)
            if (all(type(value) is int and value >= 0 for value in (incoming, outgoing, total))
                    and incoming + outgoing == total):
                entry.update(input_tokens=incoming, output_tokens=outgoing, total_tokens=total)
        return response

    def metered_search(*search_args):
        rows = retriever.search(*search_args)
        active_trace.get()["retrieved"] = [
            {key: row[key] for key in ("id", "reranker") if key in row} for row in rows
        ]
        return rows

    try:
        if retriever.mode != "milvus_embedding" or not retriever.reranker or retriever.local_only_count:
            parser.error("Require Milvus embeddings, a DashScope reranker, and a cloud-enabled test corpus.")
        workflow = build_workflow(None, SimpleNamespace(search=metered_search), None)
        with patch.object(Model, "call", metered_call), ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(collect, case, MODES[(i + offset) % len(MODES)], workflow, model)
                for i, case in enumerate(items) for offset in range(len(MODES))
            ]
            results = []
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results.append(row)
                if completed % 20 == 0 or not row["valid"]:
                    print(f"{completed}/{len(futures)} requests completed. {row['id']} {row['mode']} valid={row['valid']}",
                          flush=True)
    finally:
        if retriever.client:
            retriever.client.close()
    results.sort(key=lambda row: (row["id"], MODES.index(row["mode"])))
    report = {
        "dataset": str(args.dataset), "question_count": len(items), "chat_model": model.name,
        "embedding_model": retriever.embedding_id, "reranker_model": retriever.reranker.name,
        "knowledge_chunks": len(retriever.rows), "history": "empty", "intent": "qa", "top_k": 3,
        "summary": summarize(results), "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if any(not row["valid"] for row in results):
        raise SystemExit("Incomplete token telemetry. See the per-case records.")


if __name__ == "__main__":
    main()
