"""
Inter-annotator agreement and system scoring (PHASE 4-3 of the ACM upgrade plan).

* :func:`krippendorff_alpha` — Krippendorff's alpha for ordinal/interval data
  (interval difference metric; appropriate for the near-equidistant Likert
  scales used here). Pure Python, no external dependencies.
* :func:`compute_system_scores` — after annotation, de-blinds via the key file,
  computes mean score per system per dimension, agreement across annotators,
  and a Wilcoxon signed-rank test of TreeRAG-Beam vs each other system.

Expected annotations CSV columns:
    task_id, annotator_id, faithfulness, relevance, citation_quality
(Multiple rows per task — one per annotator.)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = ["faithfulness", "relevance", "citation_quality"]


def krippendorff_alpha(annotations: Dict[Any, Dict[Any, float]]) -> float:
    """Krippendorff's alpha (interval metric) for ``{annotator: {unit: score}}``.

    Returns 1.0 for perfect agreement, ~0 for chance, negative for systematic
    disagreement. Units rated by fewer than 2 annotators are ignored.
    """
    # Gather ratings per unit.
    by_unit: Dict[Any, List[float]] = defaultdict(list)
    for _annotator, ratings in annotations.items():
        for unit, score in ratings.items():
            if score is None:
                continue
            try:
                by_unit[unit].append(float(score))
            except (TypeError, ValueError):
                continue

    pairable = {u: v for u, v in by_unit.items() if len(v) >= 2}
    if not pairable:
        return float("nan")

    def delta2(a, b):
        return (a - b) ** 2

    # Observed disagreement.
    n = 0
    do_sum = 0.0
    all_values: List[float] = []
    for _u, vals in pairable.items():
        m = len(vals)
        n += m
        all_values.extend(vals)
        pair_sum = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    pair_sum += delta2(vals[i], vals[j])
        do_sum += pair_sum / (m - 1)
    Do = do_sum / n

    # Expected disagreement from global marginals.
    de_sum = 0.0
    for a in all_values:
        for b in all_values:
            de_sum += delta2(a, b)
    De = de_sum / (n * (n - 1)) if n > 1 else 0.0

    if De == 0:
        return 1.0 if Do == 0 else 0.0
    return 1.0 - (Do / De)


def _load_annotations(annotations_path: str):
    """Return (rows, per_dim_annotator_unit) from a completed annotations CSV."""
    rows: List[Dict[str, str]] = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_system_scores(
    annotations_path: str,
    key_path: str,
    treerag_system: str = "treerag_beam",
) -> Dict[str, Any]:
    """De-blind, score per system/dimension, agreement, and Wilcoxon tests."""
    rows = _load_annotations(annotations_path)
    with open(key_path, "r", encoding="utf-8") as f:
        key = json.load(f)

    # system -> dimension -> list of scores ; agreement: dim -> {annotator:{task:score}}
    sys_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    agree: Dict[str, Dict[str, Dict[str, float]]] = {d: defaultdict(dict) for d in DIMENSIONS}
    # For Wilcoxon: system -> task_id -> dimension -> mean score
    per_task: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in rows:
        task_id = row.get("task_id", "")
        annotator = row.get("annotator_id", "a1")
        system = key.get(task_id)
        if not system:
            continue
        for dim in DIMENSIONS:
            raw = row.get(dim, "")
            if raw is None or str(raw).strip() == "":
                continue
            try:
                score = float(raw)
            except ValueError:
                continue
            sys_scores[system][dim].append(score)
            agree[dim][annotator][task_id] = score
            per_task[system][task_id][dim].append(score)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    mean_scores = {
        s: {d: mean(sys_scores[s][d]) for d in DIMENSIONS} for s in sys_scores
    }
    alphas = {d: krippendorff_alpha(agree[d]) for d in DIMENSIONS}

    # Wilcoxon: pair systems by question_id (via task->question is not in key;
    # we pair by task ordering within system using per-task means).
    wilcoxon = {}
    try:
        from benchmarks.metrics.statistical_tests import StatisticalTests

        st = StatisticalTests()
        for d in DIMENSIONS:
            tree_vals = [mean(v[d]) for v in per_task.get(treerag_system, {}).values() if v.get(d)]
            for s in sys_scores:
                if s == treerag_system:
                    continue
                other_vals = [mean(v[d]) for v in per_task.get(s, {}).values() if v.get(d)]
                k = min(len(tree_vals), len(other_vals))
                if k >= 2:
                    res = st.wilcoxon_signed_rank(tree_vals[:k], other_vals[:k])
                    wilcoxon["{0}_vs_{1}".format(d, s)] = {
                        "p_value": getattr(res, "p_value", None),
                        "significant": getattr(res, "significant", None),
                    }
    except Exception as exc:  # pragma: no cover
        wilcoxon["error"] = str(exc)

    result = {
        "mean_scores": mean_scores,
        "krippendorff_alpha": alphas,
        "wilcoxon": wilcoxon,
        "n_rows": len(rows),
    }
    _print_scores(result)
    return result


def _print_scores(result: Dict[str, Any]) -> None:
    print("\n=== Human Evaluation — System Scores ===")
    header = "{0:<16}".format("System") + "".join("{0:>16}".format(d) for d in DIMENSIONS)
    print(header)
    print("-" * len(header))
    for s, dims in result["mean_scores"].items():
        line = "{0:<16}".format(s) + "".join("{0:>16.2f}".format(dims[d]) for d in DIMENSIONS)
        print(line)
    print("\nKrippendorff's alpha (inter-annotator agreement):")
    for d, a in result["krippendorff_alpha"].items():
        print("  {0:<16} {1:.3f}".format(d, a))
    if result["wilcoxon"]:
        print("\nWilcoxon signed-rank (TreeRAG-Beam vs others):")
        for k, v in result["wilcoxon"].items():
            if isinstance(v, dict) and "p_value" in v and v["p_value"] is not None:
                star = " *" if v.get("significant") else ""
                print("  {0:<28} p={1:.4f}{2}".format(k, v["p_value"], star))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute agreement + system scores (PHASE 4-3)")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--treerag", default="treerag_beam")
    args = parser.parse_args(argv)
    compute_system_scores(args.annotations, args.key, args.treerag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
