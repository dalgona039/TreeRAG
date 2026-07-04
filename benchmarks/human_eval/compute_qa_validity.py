"""
Compute validity statistics for qa_verification_tasks.csv after human annotation.

Metrics:
  - Validity rates: % answerable, % correct, % grounded_in_source
  - Cohen's κ between two annotators (if columns annotator_a / annotator_b exist)
  - Or single-annotator summary if only one rater

Usage:
  python benchmarks/human_eval/compute_qa_validity.py \
      --input benchmarks/human_eval/qa_verification_tasks.csv \
      --annotator-cols "annotator_a,annotator_b"  # optional: 2nd rater columns
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABEL_COLS = ["answerable(1/0)", "answer_correct(1/0)", "grounded_in_source(1/0)"]
_ROOT = Path(__file__).resolve().parents[2]


def _parse_binary(val: str) -> int | None:
    v = str(val).strip()
    if v in ("1", "1.0", "yes", "y", "true"): return 1
    if v in ("0", "0.0", "no", "n", "false"):  return 0
    return None


def _cohen_kappa(a_vals: list[int], b_vals: list[int]) -> float:
    """Cohen's kappa for binary agreement between two annotators."""
    n = len(a_vals)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(a_vals, b_vals)) / n  # observed agreement
    pa = sum(a_vals) / n
    pb = sum(b_vals) / n
    pe = pa * pb + (1 - pa) * (1 - pb)   # expected by chance
    if abs(1 - pe) < 1e-9:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(_ROOT / "benchmarks/human_eval/qa_verification_tasks.csv"))
    parser.add_argument("--output", default=str(_ROOT / "data/benchmark_reports/qa_validity_stats.json"))
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    if not rows:
        print("CSV is empty or could not be read.")
        return

    stats: dict = {"n_questions": len(rows)}
    for col in LABEL_COLS:
        vals = [_parse_binary(r.get(col, "")) for r in rows]
        valid = [v for v in vals if v is not None]
        if not valid:
            stats[col] = {"n_annotated": 0, "rate": None}
        else:
            stats[col] = {
                "n_annotated": len(valid),
                "n_positive": sum(valid),
                "rate": round(sum(valid) / len(valid), 4),
            }

    # Pairwise Cohen's κ if annotator_a and annotator_b columns exist
    if "annotator_a" in rows[0] and "annotator_b" in rows[0]:
        a_vals = [_parse_binary(r.get("annotator_a", "")) for r in rows]
        b_vals = [_parse_binary(r.get("annotator_b", "")) for r in rows]
        paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        if paired:
            kappa = _cohen_kappa([p[0] for p in paired], [p[1] for p in paired])
            stats["cohen_kappa"] = round(kappa, 4)
            stats["n_paired"] = len(paired)

    print("=== QA Validity Statistics ===")
    for col in LABEL_COLS:
        s = stats.get(col, {})
        rate = s.get("rate")
        n = s.get("n_annotated", 0)
        print(f"  {col}: {rate*100:.1f}% ({n} annotated)" if rate is not None
              else f"  {col}: not annotated")
    if "cohen_kappa" in stats:
        print(f"  Cohen's κ: {stats['cohen_kappa']} (n={stats['n_paired']})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
