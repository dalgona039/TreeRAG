"""
Evaluate hallucination detector (5-signal vs 6-signal).

Retrieves source nodes from the actual document index for a sample of answers
(poor: llm_judge ≤ 0.4 | good: llm_judge > 0.7), runs both detector versions,
and reports:
  - Mean confidence on poor vs good answers (both detectors)
  - Delta confidence poor−good (signal that detector discriminates)
  - Spearman ρ between detector confidence and LLM-Judge (full set, no nodes)

Usage:
  python scripts/eval_detector.py --signals 5
  python scripts/eval_detector.py --signals 6
  python scripts/eval_detector.py --both
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "data/benchmark_reports/online_local_llama_general_v3_n100.json"
BENCH_FILE   = ROOT / "benchmarks/datasets/full_benchmark.json"


def _load_source_nodes(doc_id: str, question: str, top_k: int = 3):
    """BM25-retrieve source nodes from the document index for a question."""
    try:
        from src.core.bm25_baseline import BM25Retriever
        from src.config import Config
        index_path = Path(Config.INDEX_DIR) / doc_id
        if not index_path.exists():
            # Try without .json extension
            index_path = Path(Config.INDEX_DIR) / (doc_id.replace(".json", "") + ".json")
        if not index_path.exists():
            return []
        index = json.load(open(index_path, encoding="utf-8"))
        retriever = BM25Retriever(index)
        return retriever.retrieve(question, top_k=top_k)
    except Exception as exc:
        return []


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0

    def rank(lst):
        si = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        for rank, idx in enumerate(si, 1):
            r[idx] = float(rank)
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) *
           sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else 0.0


def run(signals: int, results_path: Path, sample_size: int = 30, seed: int = 42) -> dict:
    from src.utils.hallucination_detector import HallucinationDetector

    use_sem = (signals == 6)
    det = HallucinationDetector(use_semantic=use_sem)

    data  = json.load(open(results_path, encoding="utf-8"))
    bench = json.load(open(BENCH_FILE, encoding="utf-8"))
    qmap  = {q["question_id"]: q for q in bench.get("questions", bench if isinstance(bench, list) else [])}

    # Flatten rows
    rows_flat = []
    for sys_name, rows in data.get("per_question", {}).items():
        for r in rows:
            if r.get("llm_judge") is not None and r.get("answer"):
                rows_flat.append({**r, "system": sys_name})

    poor = [r for r in rows_flat if r["llm_judge"] <= 0.4]
    good = [r for r in rows_flat if r["llm_judge"] > 0.7]

    rng = random.Random(seed)
    sample_poor = rng.sample(poor, min(sample_size, len(poor)))
    sample_good = rng.sample(good, min(sample_size, len(good)))

    print(f"  [{signals}-signal] Evaluating {len(sample_poor)} poor + {len(sample_good)} good answers...")

    def _score_sample(rows_subset):
        confs = []
        for row in rows_subset:
            qid = row.get("question_id", "")
            q   = qmap.get(qid, {}).get("question", "")
            nodes = _load_source_nodes(row["document_id"], q)
            result = det.detect(row["answer"], nodes)
            confs.append(result["overall_confidence"])
        return confs

    confs_poor = _score_sample(sample_poor)
    confs_good = _score_sample(sample_good)

    # Full-dataset Spearman (no nodes — both detectors degrade equally here,
    # but we include it for completeness)
    all_confs, all_judges = [], []
    for row in rows_flat:
        result = det.detect(row["answer"], [])
        all_confs.append(result["overall_confidence"])
        all_judges.append(float(row["llm_judge"]))

    import statistics
    out = {
        "signals": signals,
        "n_poor_sample": len(confs_poor),
        "n_good_sample": len(confs_good),
        "mean_conf_poor": round(statistics.mean(confs_poor), 4) if confs_poor else None,
        "mean_conf_good": round(statistics.mean(confs_good), 4) if confs_good else None,
        "delta_poor_minus_good": round(
            (statistics.mean(confs_poor) - statistics.mean(confs_good))
            if confs_poor and confs_good else 0.0, 4
        ),
        "spearman_rho_no_nodes": round(_spearman(all_confs, all_judges), 4),
        "n_total": len(all_confs),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=int, choices=[5, 6], default=6)
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"Results file not found: {results_path}")

    if args.both:
        r5 = run(5, results_path, args.sample_size, args.seed)
        r6 = run(6, results_path, args.sample_size, args.seed)
        print("\n=== 5-signal vs 6-signal comparison ===")
        for k in r5:
            if k == "signals":
                continue
            v5, v6 = r5[k], r6[k]
            if isinstance(v5, float) and isinstance(v6, float):
                delta = f"  Δ={v6 - v5:+.4f}"
            else:
                delta = ""
            print(f"  {k:45s}: 5→{v5}  6→{v6}{delta}")
        out_path = ROOT / "data/benchmark_reports/detector_eval_comparison.json"
        json.dump({"5_signals": r5, "6_signals": r6}, open(out_path, "w"), indent=2)
        print(f"\nSaved → {out_path}")
    else:
        r = run(args.signals, results_path, args.sample_size, args.seed)
        for k, v in r.items():
            print(f"  {k}: {v}")
        out_path = ROOT / f"data/benchmark_reports/detector_eval_{args.signals}signals.json"
        json.dump(r, open(out_path, "w"), indent=2)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
