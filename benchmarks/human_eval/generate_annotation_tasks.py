"""
Annotation task generator (PHASE 4-2 of the ACM upgrade plan).

Selects a stratified subset of questions from an evaluation report and emits a
*blinded* annotation CSV (one row per system per question, system identity
removed and rows shuffled) plus a separate ``annotation_key.json`` mapping
``task_id -> system_name`` for de-blinding after annotation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = _PROJECT_ROOT / "data" / "benchmark_reports" / "evaluation_latest.json"
DEFAULT_DATASET = _PROJECT_ROOT / "benchmarks" / "datasets" / "full_benchmark.json"
DEFAULT_OUTPUT = _PROJECT_ROOT / "benchmarks" / "human_eval" / "annotation_tasks.csv"

CSV_COLUMNS = [
    "task_id",
    "question_id",
    "question",
    "source_excerpt",
    "answer",
    "faithfulness",
    "relevance",
    "citation_quality",
    "notes",
]


def _question_meta(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {q["question_id"]: q for q in dataset.get("questions", [])}


def _stratified_sample(
    meta: Dict[str, Dict[str, Any]], available_qids: List[str], n: int, seed: int
) -> List[str]:
    """Sample ``n`` qids stratified by (difficulty, category), deterministic."""
    strata: Dict[Any, List[str]] = defaultdict(list)
    for qid in available_qids:
        m = meta.get(qid, {})
        strata[(m.get("difficulty", "?"), m.get("category", "?"))].append(qid)

    rng = random.Random(seed)
    for k in strata:
        strata[k].sort()
        rng.shuffle(strata[k])

    selected: List[str] = []
    keys = sorted(strata.keys())
    # Round-robin across strata until we reach n.
    idx = 0
    while len(selected) < n and any(strata.values()):
        k = keys[idx % len(keys)]
        if strata[k]:
            selected.append(strata[k].pop())
        idx += 1
        if idx > len(keys) * (n + 5):
            break
    return selected[:n]


def generate_annotation_tasks(
    evaluation_results_path: str,
    n_questions: int = 50,
    systems: Optional[List[str]] = None,
    output_path: str = str(DEFAULT_OUTPUT),
    dataset_path: str = str(DEFAULT_DATASET),
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a blinded annotation CSV + key file. Returns a small summary."""
    systems = systems or ["raptor", "flatrag", "treerag_beam"]

    with open(evaluation_results_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    meta = _question_meta(dataset)

    pq = report.get("per_question", {})
    answer_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sysname in systems:
        answer_lookup[sysname] = {r["question_id"]: r for r in pq.get(sysname, [])}

    # Questions answered by every requested system.
    common_qids = None
    for sysname in systems:
        ids = set(answer_lookup[sysname].keys())
        common_qids = ids if common_qids is None else (common_qids & ids)
    common_qids = sorted(common_qids or [])

    selected = _stratified_sample(meta, common_qids, n_questions, seed)

    rows: List[Dict[str, Any]] = []
    key: Dict[str, str] = {}
    counter = 1
    for qid in selected:
        m = meta.get(qid, {})
        for sysname in systems:
            row_ans = answer_lookup[sysname].get(qid, {})
            task_id = "T{0:04d}".format(counter)
            counter += 1
            rows.append(
                {
                    "task_id": task_id,
                    "question_id": qid,
                    "question": m.get("question", ""),
                    "source_excerpt": m.get("expected_answer_hint", ""),
                    "answer": row_ans.get("answer", ""),
                    "faithfulness": "",
                    "relevance": "",
                    "citation_quality": "",
                    "notes": "",
                }
            )
            key[task_id] = sysname

    # Shuffle so consecutive rows are not the same system (blinding).
    rng = random.Random(seed + 1)
    rng.shuffle(rows)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    key_path = out.parent / "annotation_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    summary = {
        "n_questions": len(selected),
        "systems": systems,
        "total_rows": len(rows),
        "csv": str(out),
        "key": str(key_path),
    }
    print("Annotation tasks: {0} questions x {1} systems = {2} rows".format(
        len(selected), len(systems), len(rows)))
    print("  CSV : {0}".format(out))
    print("  Key : {0}".format(key_path))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate annotation tasks (PHASE 4-2)")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--systems", default="raptor,flatrag,treerag_beam")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    generate_annotation_tasks(
        args.report,
        n_questions=args.n_questions,
        systems=[s.strip() for s in args.systems.split(",") if s.strip()],
        output_path=args.output,
        dataset_path=args.dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
