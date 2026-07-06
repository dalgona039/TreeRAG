#!/usr/bin/env python3
"""
Diagnose whether the response-cache bug (traversal_algorithm missing from the
cache key in src/core/reasoner.py, fixed for the Task-4 auto-policy re-run)
also contaminated earlier report JSONs — i.e. whether treerag_dfs and
treerag_beam answers for the same question are suspiciously identical.

Same diagnostic the auto-policy re-run used (42/50 -> 2/50 after the fix):
for each report JSON, pair up treerag_dfs and treerag_beam answers by
question_id and report what fraction are exactly identical (normalized
whitespace) and what fraction share a long identical prefix (>=80 chars),
which is a softer signal of "same cached generation, different retrieval
stats bolted on".

Usage:
    python scripts/check_cache_contamination.py
    python scripts/check_cache_contamination.py path/to/report1.json path/to/report2.json

With no arguments, scans data/benchmark_reports/ for the files most likely to
have been used for Tables 3/8/9/10/12 (online_local_llama_*.json and
exp2_multihop_*.json), skipping *.ckpt.json checkpoints.

A clean (uncontaminated) run should look like the auto-policy "_fixed" file:
only a few percent of coincidental exact matches. Contamination looks like
the pre-fix auto-policy file: ~40-80% identical.
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "benchmark_reports"

PAIR_A = "treerag_dfs"
PAIR_B = "treerag_beam"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_pair(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"could not parse JSON ({e})"

    pq = data.get("per_question")
    if not isinstance(pq, dict):
        return None, "no per_question dict"
    if PAIR_A not in pq or PAIR_B not in pq:
        return None, f"missing {PAIR_A!r} or {PAIR_B!r} in per_question"

    def index_by_qid(records):
        out = {}
        for r in records:
            qid = r.get("question_id") or r.get("qid") or r.get("id")
            if qid is not None:
                out[qid] = r
        return out

    a = index_by_qid(pq[PAIR_A])
    b = index_by_qid(pq[PAIR_B])
    common = sorted(set(a) & set(b))
    if not common:
        return None, "no overlapping question_ids between dfs and beam"

    exact = 0
    prefix_match = 0
    examples = []
    for qid in common:
        ans_a = normalize(a[qid].get("answer", ""))
        ans_b = normalize(b[qid].get("answer", ""))
        if not ans_a and not ans_b:
            continue
        if ans_a == ans_b:
            exact += 1
            if len(examples) < 3:
                examples.append((qid, ans_a[:100]))
        elif len(ans_a) >= 80 and ans_a[:80] == ans_b[:80]:
            prefix_match += 1

    n = len(common)
    return {
        "n_common": n,
        "exact_match": exact,
        "exact_match_pct": 100.0 * exact / n if n else 0.0,
        "long_prefix_match": prefix_match,
        "long_prefix_match_pct": 100.0 * prefix_match / n if n else 0.0,
        "examples": examples,
    }, None


def default_targets():
    patterns = [
        "online_local_llama_*.json",
        "exp2_multihop_*.json",
    ]
    files = []
    for pat in patterns:
        for f in sorted(glob.glob(str(REPORT_DIR / pat))):
            if f.endswith(".ckpt.json"):
                continue
            files.append(Path(f))
    return files


def main():
    targets = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else default_targets()
    if not targets:
        print(f"No candidate report JSONs found under {REPORT_DIR}")
        return 1

    print(f"{'file':55s} {'n':>5s} {'exact%':>8s} {'prefix%':>8s}  verdict")
    print("-" * 95)
    for path in targets:
        result, err = load_pair(path)
        if err:
            print(f"{path.name:55s} {'--':>5s} {'--':>8s} {'--':>8s}  skipped ({err})")
            continue
        pct = result["exact_match_pct"]
        verdict = "CLEAN (coincidental ties only)" if pct < 10 else (
            "SUSPICIOUS (check further)" if pct < 25 else
            "LIKELY CONTAMINATED"
        )
        print(f"{path.name:55s} {result['n_common']:5d} {pct:7.1f}% "
              f"{result['long_prefix_match_pct']:7.1f}%  {verdict}")
        if pct >= 10 and result["examples"]:
            for qid, snippet in result["examples"]:
                print(f"    e.g. {qid}: {snippet!r}...")
    print()
    print("Reference points: the auto-policy re-run measured 42/50 = 84.0% before "
          "the traversal_algorithm cache-key fix, and 2/50 = 4.0% after.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
