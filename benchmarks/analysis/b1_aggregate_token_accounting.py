"""B1 — aggregate raw per-call token-accounting records into a per-system table.

The instrumentation (src/utils/llm_accounting.py) tags each LLM call with a
*pipeline stage* (traversal_dfs, traversal_beam, generation, generation_baseline,
judge, raptor_index_summarization), not a system name, because generation/judge are
shared code paths hit by multiple systems. Systems run strictly sequentially
(benchmarks/run_real_evaluation.py's evaluate() loop finishes one system's 100
questions before starting the next), so call records preserve enough order
information to reconstruct the per-system split:

  - raptor_index_summarization: one-time indexing calls, pulled out separately
    (not a per-query cost) regardless of position.
  - The remaining records split into BM25 / Dense / FlatRAG / RAPTOR blocks of
    exactly 200 records each (100 generation_baseline + 100 judge, run order),
    followed by a DFS block and a Beam block.
  - Within the DFS/Beam tail, the boundary is the first "traversal_beam" record
    (Beam's traversal calls only start once DFS's 100 questions are entirely done).
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

BASELINE_SYSTEMS = ["BM25", "Dense Retrieval", "FlatRAG", "RAPTOR"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(records, n_queries):
    calls = len(records)
    prompt = [r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]
    completion = [r["completion_tokens"] for r in records if r["completion_tokens"] is not None]
    total = [r["total_tokens"] for r in records if r["total_tokens"] is not None]
    latency = [r["latency_s"] for r in records]
    return {
        "n_queries": n_queries,
        "llm_calls": calls,
        "calls_per_query": calls / n_queries if n_queries else 0,
        "prompt_tok_per_query": sum(prompt) / n_queries if n_queries else 0,
        "completion_tok_per_query": sum(completion) / n_queries if n_queries else 0,
        "total_tok_per_query": sum(total) / n_queries if n_queries else 0,
        "total_tok_sum": sum(total),
        "mean_call_latency_s": stats.mean(latency) if latency else 0,
        "sum_latency_s": sum(latency),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-accounting", required=True)
    ap.add_argument("--dense-fix-accounting", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    main_acct = load(args.main_accounting)
    dense_acct = load(args.dense_fix_accounting)

    raptor_index = [r for r in main_acct if r["stage"] == "raptor_index_summarization"]
    rest = [r for r in main_acct if r["stage"] != "raptor_index_summarization"]

    # First 800 non-indexing records: 4 baseline systems x 200 (100 gen + 100 judge), in run order.
    blocks = {}
    for i, name in enumerate(BASELINE_SYSTEMS):
        blocks[name] = rest[i * 200:(i + 1) * 200]
    tail = rest[800:]

    beam_start = next((i for i, r in enumerate(tail) if r["stage"] == "traversal_beam"), len(tail))
    dfs_block = tail[:beam_start]
    beam_block = tail[beam_start:]

    # Sanity check: verify each baseline block is purely generation_baseline+judge.
    diagnostics = {}
    for name, block in blocks.items():
        stages = sorted(set(r["stage"] for r in block))
        diagnostics[name] = {"n": len(block), "stages": stages}
    diagnostics["TreeRAG-DFS"] = {"n": len(dfs_block), "stages": sorted(set(r["stage"] for r in dfs_block))}
    diagnostics["TreeRAG-Beam"] = {"n": len(beam_block), "stages": sorted(set(r["stage"] for r in beam_block))}
    diagnostics["raptor_indexing_calls"] = len(raptor_index)
    diagnostics["unknown_stage_records"] = [r for r in main_acct if r["stage"] == "unknown"]

    results = {
        "BM25": summarize(blocks["BM25"], 100),
        "Dense Retrieval": summarize(dense_acct, 100),  # corrected rerun, not the broken main-run block
        "FlatRAG": summarize(blocks["FlatRAG"], 100),
        "RAPTOR (query-time only)": summarize(blocks["RAPTOR"], 100),
        "RAPTOR (one-time indexing)": summarize(raptor_index, 25),  # 25 unique docs in the n=100 sample
        "PageTree-RAG (DFS)": summarize(dfs_block, 100),
        "PageTree-RAG (Beam)": summarize(beam_block, 100),
    }

    out = {"diagnostics": diagnostics, "results": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
