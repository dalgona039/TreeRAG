"""B4 — FlatRAG 60/25/15 weight sensitivity sweep, retrieval-only (no LLM calls).

FlatRAGBaseline's BM25/"semantic"(Jaccard word-overlap)/structural(depth-prior) scorers
are pure algorithms with no embedding model or LLM in the loop, so retrieval quality
under different weight combinations can be measured directly against the gold
`expected_sections` (the same field citation_metrics.py uses) without generating any
answers — sidesteps needing the local Ollama server, which is occupied by the B1 rerun.

Usage:
    python -m benchmarks.analysis.b4_flatrag_weight_sweep \
        --dataset benchmarks/datasets/full_benchmark.json --limit 100 --seed 42 \
        --out docs/paper_revision/B4_weight_sweep_n100.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

from src.core.flat_rag_baseline import FlatRAGBaseline

WEIGHT_CONFIGS = {
    "current_60_25_15": (0.60, 0.25, 0.15),
    "pure_bm25_100_0_0": (1.00, 0.00, 0.00),
    "pure_semantic_0_100_0": (0.00, 1.00, 0.00),
    "pure_structural_0_0_100": (0.00, 0.00, 1.00),
    "equal_33_33_33": (1 / 3, 1 / 3, 1 / 3),
    "bm25_heavy_80_10_10": (0.80, 0.10, 0.10),
    "semantic_heavy_20_70_10": (0.20, 0.70, 0.10),
}


def _match(retrieved_id: str, expected_id: str) -> bool:
    r, e = retrieved_id.strip(), expected_id.strip()
    if not r or not e:
        return False
    return r == e or r.startswith(e + ".") or e.startswith(r + ".")


def score_config(baseline: FlatRAGBaseline, query: str, weights, top_k: int = 5):
    w_bm25, w_sem, w_struct = weights
    bm25_results = baseline.bm25.search(query, top_k=20)
    max_bm25 = max((score for _, score in bm25_results), default=1.0)
    scored = []
    for doc_id, bm25_score in bm25_results:
        doc = baseline.documents[doc_id]
        semantic_score = baseline.semantic_ranker.score(query, doc["text"])
        structural_score = baseline.structural_ranker.score(doc, doc["depth"])
        normalized_bm25 = (bm25_score / max_bm25) if max_bm25 > 0 else 0
        combined = w_bm25 * normalized_bm25 + w_sem * semantic_score + w_struct * structural_score
        scored.append((doc_id, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [baseline.documents[doc_id]["id"] for doc_id, _ in scored[:top_k]]
    return top_ids


def section_f1(retrieved_ids, expected_sections):
    if not expected_sections:
        return None
    r_ids = [x for x in retrieved_ids if x]
    e_ids = [str(e) for e in expected_sections if e]
    if not e_ids:
        return None
    if not r_ids:
        return 0.0
    p_hits = sum(1 for r in r_ids if any(_match(r, e) for e in e_ids))
    precision = p_hits / len(r_ids)
    r_hits = sum(1 for e in e_ids if any(_match(r, e) for r in r_ids))
    recall = r_hits / len(e_ids)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    qs = list(dataset["questions"])
    if args.seed:
        random.Random(args.seed).shuffle(qs)
    qs = qs[: args.limit]

    baseline_cache = {}
    rows = []
    per_config_f1 = defaultdict(list)
    per_config_topsets = defaultdict(dict)  # config -> {qid: set(ids)}

    for q in qs:
        doc_id = q["document_id"]
        if doc_id not in baseline_cache:
            baseline_cache[doc_id] = FlatRAGBaseline([doc_id])
        baseline = baseline_cache[doc_id]
        expected = q.get("expected_sections") or []
        for cfg_name, weights in WEIGHT_CONFIGS.items():
            top_ids = score_config(baseline, q["question"], weights)
            f1 = section_f1(top_ids, expected)
            per_config_topsets[cfg_name][q["question_id"]] = set(top_ids)
            if f1 is not None:
                per_config_f1[cfg_name].append(f1)
            rows.append({
                "question_id": q["question_id"], "document_id": doc_id,
                "config": cfg_name, "section_f1": f1,
                "top_ids": "|".join(top_ids),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}\n")

    print("=== Mean section-F1 (retrieval-only proxy) per weight config ===")
    for cfg_name in WEIGHT_CONFIGS:
        vals = per_config_f1[cfg_name]
        if vals:
            print(f"  {cfg_name:28s} n={len(vals):3d}  mean_F1={stats.mean(vals):.4f}  std={stats.pstdev(vals):.4f}")
        else:
            print(f"  {cfg_name:28s} n=0 (no questions had expected_sections)")

    print("\n=== Jaccard overlap of top-5 retrieved ID sets vs. current_60_25_15 ===")
    base_sets = per_config_topsets["current_60_25_15"]
    for cfg_name in WEIGHT_CONFIGS:
        if cfg_name == "current_60_25_15":
            continue
        overlaps = []
        for qid, base_set in base_sets.items():
            other_set = per_config_topsets[cfg_name].get(qid, set())
            union = base_set | other_set
            inter = base_set & other_set
            overlaps.append(len(inter) / len(union) if union else 1.0)
        print(f"  {cfg_name:28s} mean_jaccard_vs_current={stats.mean(overlaps):.4f}")


if __name__ == "__main__":
    main()
