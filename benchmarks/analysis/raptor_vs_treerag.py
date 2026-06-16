"""
RAPTOR vs TreeRAG differentiation analysis (PHASE 1-5 of the ACM upgrade plan).

Given an evaluation report (from ``run_real_evaluation.py``) and the source
benchmark dataset (for question categories), computes the head-to-head evidence
that goes into the paper's Discussion section:

* Mean ROUGE-L difference (TreeRAG - RAPTOR)
* Mean latency difference
* Win rate: % of questions where TreeRAG beats RAPTOR on ROUGE-L
* Category breakdown: factual / multi_hop / comparative win rates
* Page-citation availability: % of answers containing ``[doc, p.X]`` markers
  (a TreeRAG capability; RAPTOR's bottom-up summaries are not page-traceable)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = _PROJECT_ROOT / "data" / "benchmark_reports" / "evaluation_latest.json"
DEFAULT_DATASET = _PROJECT_ROOT / "benchmarks" / "datasets" / "full_benchmark.json"

# Matches page citations like "[문서A, p.5]", "[doc, p. 12-14]".
_CITATION_RE = re.compile(r"\[[^\]]*p\.\s*\d+", re.IGNORECASE)


def has_page_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer or ""))


def citation_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    cited = sum(1 for r in rows if has_page_citation(r.get("answer", "")))
    return cited / len(rows)


def _by_qid(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["question_id"]: r for r in rows}


def _category_map(dataset: Dict[str, Any]) -> Dict[str, str]:
    return {q["question_id"]: q.get("category", "unknown") for q in dataset.get("questions", [])}


def analyze(
    report: Dict[str, Any],
    dataset: Optional[Dict[str, Any]] = None,
    treerag_system: str = "treerag_beam",
    raptor_system: str = "raptor",
) -> Dict[str, Any]:
    pq = report.get("per_question", {})
    if treerag_system not in pq or raptor_system not in pq:
        raise ValueError(
            "Report must contain both '{0}' and '{1}' systems.".format(
                treerag_system, raptor_system
            )
        )

    tree = _by_qid(pq[treerag_system])
    rapt = _by_qid(pq[raptor_system])
    common = [qid for qid in tree if qid in rapt]

    cat_map = _category_map(dataset) if dataset else {}

    wins = 0
    rouge_diffs: List[float] = []
    latency_diffs: List[float] = []
    cat_totals: Dict[str, int] = {}
    cat_wins: Dict[str, int] = {}

    for qid in common:
        t, r = tree[qid], rapt[qid]
        d_rouge = t["rouge_l"] - r["rouge_l"]
        rouge_diffs.append(d_rouge)
        latency_diffs.append(t.get("latency", 0.0) - r.get("latency", 0.0))
        won = t["rouge_l"] > r["rouge_l"]
        if won:
            wins += 1
        cat = cat_map.get(qid, "unknown")
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        cat_wins[cat] = cat_wins.get(cat, 0) + (1 if won else 0)

    n = len(common) or 1
    result = {
        "n_questions": len(common),
        "mean_rouge_l_diff": sum(rouge_diffs) / n,
        "mean_latency_diff": sum(latency_diffs) / n,
        "treerag_win_rate": wins / n,
        "category_win_rate": {
            c: (cat_wins[c] / cat_totals[c] if cat_totals[c] else 0.0)
            for c in sorted(cat_totals)
        },
        "citation_rate": {
            treerag_system: citation_rate(pq[treerag_system]),
            raptor_system: citation_rate(pq[raptor_system]),
        },
    }
    return result


def print_report(result: Dict[str, Any]) -> None:
    print("\n=== RAPTOR vs TreeRAG — Differentiation Analysis ===")
    print("Questions compared      : {0}".format(result["n_questions"]))
    print("Mean ROUGE-L diff (T-R) : {0:+.3f}".format(result["mean_rouge_l_diff"]))
    print("Mean latency diff (T-R) : {0:+.4f}s".format(result["mean_latency_diff"]))
    print("TreeRAG win rate        : {0:.1%}".format(result["treerag_win_rate"]))
    print("Win rate by category:")
    for cat, wr in result["category_win_rate"].items():
        print("  {0:<12} {1:.1%}".format(cat, wr))
    print("Page-citation availability:")
    for sysname, rate in result["citation_rate"].items():
        print("  {0:<14} {1:.1%}".format(sysname, rate))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAPTOR vs TreeRAG analysis (PHASE 1-5)")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--treerag", default="treerag_beam")
    parser.add_argument("--raptor", default="raptor")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)
    dataset = None
    if args.dataset and Path(args.dataset).exists():
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    result = analyze(report, dataset, args.treerag, args.raptor)
    print_report(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n💾 Analysis → {0}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
