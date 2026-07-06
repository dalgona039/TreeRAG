#!/usr/bin/env python3
"""
Experiment 2: Multi-hop Quality

Part A  — Run 6 systems on the HotpotQA subset (local flat-tree indices).
Part B  — Break down an existing full_benchmark report by question category
           (factual / multi_hop / comparative) and compute per-category
           ROUGE-L and LLM-Judge averages.
           Then run paired t-tests for the multi_hop subset:
           TreeRAG-Beam vs each baseline, and vs TreeRAG-DFS.

Usage:
  # Smoke (Part A only, 5 questions):
  python benchmarks/run_exp2_multihop.py --smoke --limit 5

  # Full Part A (HotpotQA 20q) + Part B (existing 40Q report):
  python benchmarks/run_exp2_multihop.py

  # Specify a different existing report for Part B:
  python benchmarks/run_exp2_multihop.py \
      --existing-report data/benchmark_reports/online_local_llama_general_v2.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORT_DIR = _ROOT / "data" / "benchmark_reports"

ALL_SYSTEMS = ["bm25", "dense", "flatrag", "raptor", "treerag_dfs", "treerag_beam"]
SYSTEM_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "flatrag": "FlatRAG",
    "raptor": "RAPTOR",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}
CATEGORIES = ["factual", "multi_hop", "comparative"]


def _mean(vals: List[Any]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Part A: HotpotQA evaluation
# ---------------------------------------------------------------------------
def run_hotpotqa(n: int, seed: int, smoke: bool, limit: int = 0) -> Optional[Dict[str, Any]]:
    """Run 6 systems on HotpotQA and return per_question dict.

    Args:
        n: Number of questions to build and evaluate (default 100).
        seed: Random seed.
        smoke: If True, print sample answers and validate.
        limit: If >0, further sub-sample to this many (e.g. --limit 5 for smoke).
    """
    effective = min(limit, n) if limit else n
    print("\n" + "=" * 60)
    print(f"Part A: HotpotQA evaluation (n={effective}, seed={seed})")
    print("=" * 60)

    from benchmarks.datasets.hotpotqa_loader import build_hotpotqa_dataset
    from benchmarks.run_real_evaluation import (
        ALL_SYSTEMS as _ALL_SYS, Evaluator, SYSTEM_LABELS as _SL,
        aggregate, significance, print_table, print_significance,
        save_markdown_table,
    )
    from benchmarks.metrics.text_similarity import rouge_l_score
    from src.core.ollama_client import OllamaClient
    from src.config import set_client_override

    client = OllamaClient(model="llama3.1:8b")
    set_client_override(client)

    dataset = build_hotpotqa_dataset(n=n, seed=seed)
    questions = dataset["questions"]
    if limit and limit < len(questions):
        rng = random.Random(seed)
        rng.shuffle(questions)
        questions = questions[:limit]
    dataset = {**dataset, "questions": questions}

    print(f"Questions: {len(questions)}")

    evaluator = Evaluator(
        mode="online",
        use_llm_judge=True,
        local_judge=True,
        local_judge_model="llama3.1:8b",
        gen_backend="ollama",
        gen_model="llama3.1:8b",
    )

    per_system: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ALL_SYSTEMS}
    import time
    for system in ALL_SYSTEMS:
        print(f"\n▶ {SYSTEM_LABELS[system]} — {len(questions)} questions")
        for q in questions:
            t0 = time.perf_counter()
            try:
                answer, nodes = evaluator.run_system(
                    system, q["question"], q["document_id"]
                )
            except Exception as exc:
                print(f"   ⚠ {system} failed on {q['question_id']}: {exc}")
                answer, nodes = "", []
            latency = time.perf_counter() - t0

            expected = q.get("expected_answer_hint", "")
            ctx = " ".join(n.get("summary", n.get("title", "")) for n in nodes)
            scores = evaluator.score_answer(q["question"], ctx, answer, expected)

            if smoke:
                print(f"   Q: {q['question'][:80]}")
                print(f"   A ({len(answer)} chars, {latency:.1f}s): {answer[:200]}")
                if len(answer) < 10:
                    print("   ⛔ ANSWER TOO SHORT")

            per_system[system].append({
                "question_id": q["question_id"],
                "document_id": q["document_id"],
                "answer": answer,
                "latency": latency,
                "context_tokens": int(len(ctx) / 4),
                **scores,
            })

    agg = aggregate(per_system)
    sig = significance(per_system, ALL_SYSTEMS)
    print_table(agg, ALL_SYSTEMS)
    print_significance(sig)

    # Smoke validation
    if smoke:
        failed = []
        for s in ALL_SYSTEMS:
            rows = per_system[s]
            bad = [r for r in rows if len(r["answer"]) < 10]
            if bad:
                failed.append(f"{s}: {len(bad)} short answers")
            slow = [r for r in rows if r["latency"] < 1.0]
            if slow:
                failed.append(f"{s}: {len(slow)} fast answers (latency<1s → no LLM?)")
        if failed:
            print("\n⛔ SMOKE FAIL:")
            for msg in failed:
                print(f"  - {msg}")
            return None
        print("\n✅ SMOKE PASS (Part A)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"exp2_multihop_hotpotqa_{stamp}.json"
    md_path = REPORT_DIR / f"exp2_multihop_hotpotqa_{stamp}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": "hotpotqa", "n_questions": len(questions),
             "summary": agg, "significance": sig, "per_question": per_system},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n💾 Report → {json_path}")
    save_markdown_table(
        agg, sig, ALL_SYSTEMS, md_path,
        meta={"gen_backend": "ollama", "gen_model": "llama3.1:8b",
              "dataset": "hotpotqa", "n_questions": len(questions),
              "seed": seed, "date": stamp},
    )
    return {"per_question": per_system, "summary": agg, "significance": sig}


# ---------------------------------------------------------------------------
# Part B: Category breakdown of existing full_benchmark report
# ---------------------------------------------------------------------------
def run_category_breakdown(report_path: str) -> None:
    print("\n" + "=" * 60)
    print("Part B: Category breakdown of existing report")
    print(f"  Source: {report_path}")
    print("=" * 60)

    # Load the full_benchmark dataset for category mapping
    fb_path = _ROOT / "benchmarks" / "datasets" / "full_benchmark.json"
    with open(fb_path, encoding="utf-8") as f:
        fb = json.load(f)
    cat_map: Dict[str, str] = {q["question_id"]: q["category"] for q in fb["questions"]}

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    per_q = report.get("per_question", {})
    systems = report.get("systems", list(per_q.keys()))

    # Per-category aggregation
    cat_stats: Dict[str, Dict[str, Dict[str, float]]] = {cat: {} for cat in CATEGORIES}
    for system in systems:
        rows = per_q.get(system, [])
        by_cat: Dict[str, List] = {c: [] for c in CATEGORIES}
        for row in rows:
            cat = cat_map.get(row["question_id"])
            if cat in by_cat:
                by_cat[cat].append(row)
        for cat in CATEGORIES:
            cat_rows = by_cat[cat]
            if not cat_rows:
                cat_stats[cat][system] = {"n": 0}
                continue
            cat_stats[cat][system] = {
                "n": len(cat_rows),
                "rouge_l": _mean([r["rouge_l"] for r in cat_rows]),
                "llm_judge": _mean([r["llm_judge"] for r in cat_rows]),
            }

    # Print per-category table
    for cat in CATEGORIES:
        print(f"\n  Category: {cat.replace('_',' ').title()}")
        print(f"  {'System':<18}  {'ROUGE-L':>8}  {'LLM-Judge':>10}  {'N':>4}")
        print(f"  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*4}")
        for s in systems:
            st = cat_stats[cat].get(s, {})
            n = st.get("n", 0)
            rl = st.get("rouge_l")
            lj = st.get("llm_judge")
            rl_str = f"{rl:.3f}" if rl is not None else "  —  "
            lj_str = f"{lj:.3f}" if lj is not None else "  —  "
            print(f"  {SYSTEM_LABELS.get(s,s):<18}  {rl_str:>8}  {lj_str:>10}  {n:>4}")

    # Paired t-test on multi_hop subset
    print("\n  Paired t-test (multi_hop subset, TreeRAG-Beam vs others):")
    if "treerag_beam" in systems:
        from benchmarks.metrics.statistical_tests import StatisticalTests
        st_test = StatisticalTests()
        beam_rows = {r["question_id"]: r for r in per_q.get("treerag_beam", [])
                     if cat_map.get(r["question_id"]) == "multi_hop"}
        for metric in ("rouge_l", "llm_judge"):
            print(f"\n    Metric: {metric}")
            for s in systems:
                if s == "treerag_beam":
                    continue
                other_rows = {r["question_id"]: r for r in per_q.get(s, [])
                              if cat_map.get(r["question_id"]) == "multi_hop"}
                common = [qid for qid in beam_rows if qid in other_rows]
                if len(common) < 3:
                    print(f"    vs {SYSTEM_LABELS.get(s,s):<18}  n={len(common)} (too few)")
                    continue
                a = [beam_rows[qid][metric] for qid in common if beam_rows[qid].get(metric) is not None]
                b = [other_rows[qid][metric] for qid in common if other_rows[qid].get(metric) is not None]
                if len(a) < 3:
                    print(f"    vs {SYSTEM_LABELS.get(s,s):<18}  n={len(a)} (too few valid)")
                    continue
                res = st_test.paired_ttest(a, b)
                star = " *" if res.significant else ""
                print(f"    vs {SYSTEM_LABELS.get(s,s):<18}  p={res.p_value:.4f}"
                      f"  Δ={res.mean_difference:+.3f}  d={res.effect_size:.2f}{star}  n={len(a)}")

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "source_report": report_path,
        "category_stats": cat_stats,
    }
    json_path = REPORT_DIR / f"exp2_multihop_catbreakdown_{stamp}.json"
    md_path = REPORT_DIR / f"exp2_multihop_catbreakdown_{stamp}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON → {json_path}")

    # Markdown
    md_lines = [
        "# Experiment 2: Multi-hop Quality — Category Breakdown",
        "",
        f"Source: `{Path(report_path).name}`",
    ]
    for cat in CATEGORIES:
        md_lines += ["", f"## {cat.replace('_',' ').title()}",
                     "", "| System | ROUGE-L | LLM-Judge | N |",
                     "|--------|---------|-----------|---|"]
        for s in systems:
            st = cat_stats[cat].get(s, {})
            rl = st.get("rouge_l")
            lj = st.get("llm_judge")
            md_lines.append(
                f"| {SYSTEM_LABELS.get(s,s)} "
                f"| {rl:.3f} " if rl is not None else f"| — "
                f"| {lj:.3f} " if lj is not None else f"| — "
                f"| {st.get('n',0)} |"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"📄 Markdown → {md_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Multi-hop Quality")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: print sample answers, validate, skip full run")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of HotpotQA questions to build and evaluate (default=100)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Sub-sample to this many after building (0=use --n; useful for smoke --limit 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-partA", action="store_true",
                        help="Skip HotpotQA evaluation (Part A)")
    parser.add_argument("--skip-partB", action="store_true",
                        help="Skip category breakdown (Part B)")
    parser.add_argument(
        "--existing-report",
        default=str(REPORT_DIR / "online_local_llama_general_v2.json"),
        help="Existing full_benchmark report JSON for Part B",
    )
    args = parser.parse_args()

    if not args.skip_partA:
        result = run_hotpotqa(
            n=args.n,
            seed=args.seed,
            smoke=args.smoke,
            limit=args.limit,
        )
        if args.smoke and result is None:
            sys.exit(1)

    if not args.skip_partB:
        run_category_breakdown(args.existing_report)


if __name__ == "__main__":
    main()
