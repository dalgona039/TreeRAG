#!/usr/bin/env python3
"""Robust small-sample statistics for TreeRAG benchmarks.

Computes, per benchmark and per (TreeRAG vs baseline) comparison:
  - paired mean difference in ROUGE-L
  - bootstrap 95% CI of the mean difference (10k resamples, paired)
  - permutation (randomization) test p-value (paired sign-flip, 10k)
  - paired Cohen's d_z and its bootstrap 95% CI
  - post-hoc achieved power (paired t, alpha=.05, two-sided)
  - required n for power = 0.80 at the observed effect size

Reads existing per_question result JSONs; no external dependencies beyond
numpy / scipy. Pure re-analysis of already-collected data.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data" / "benchmark_reports"
RNG = np.random.default_rng(20260629)
B = 10000
METRIC = "rouge_l"


def load_pq(fname):
    d = json.load(open(REP / fname, encoding="utf-8"))
    return d["per_question"]


def aligned(pq, sys_a, sys_b, metric=METRIC):
    """Return paired arrays aligned by question_id."""
    a = {r["question_id"]: r.get(metric) for r in pq[sys_a]}
    b = {r["question_id"]: r.get(metric) for r in pq[sys_b]}
    ids = [q for q in a if q in b and a[q] is not None and b[q] is not None]
    xa = np.array([a[q] for q in ids], float)
    xb = np.array([b[q] for q in ids], float)
    return xa, xb


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def required_n(dz, power=0.80, alpha=0.05):
    if dz == 0:
        return float("inf")
    from scipy.stats import norm
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    n = ((za + zb) / abs(dz)) ** 2 + 1  # +1 df correction approx
    return int(np.ceil(n))


def achieved_power(dz, n, alpha=0.05):
    """Two-sided paired t-test post-hoc power via noncentral t."""
    if n < 2:
        return float("nan")
    df = n - 1
    ncp = abs(dz) * np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    # power = P(|T'| > tcrit) under noncentral t
    upper = stats.nct.sf(tcrit, df, ncp)   # 1 - cdf, numerically stable
    lower = stats.nct.cdf(-tcrit, df, ncp)
    p = float(upper + lower)
    if not np.isfinite(p):
        # large-ncp regime: effect is enormous, power saturates at ~1
        p = 1.0 if ncp > tcrit + 5 else float("nan")
    return min(p, 1.0)


def analyze(xa, xb):
    """xa = TreeRAG, xb = baseline. Positive diff = TreeRAG better."""
    diff = xa - xb
    n = len(diff)
    md = diff.mean()
    # bootstrap CI of mean diff + dz
    bs_md = np.empty(B); bs_dz = np.empty(B)
    for i in range(B):
        idx = RNG.integers(0, n, n)
        d = diff[idx]
        bs_md[i] = d.mean()
        sd = d.std(ddof=1)
        bs_dz[i] = d.mean() / sd if sd > 0 else 0.0
    ci_md = np.percentile(bs_md, [2.5, 97.5])
    ci_dz = np.percentile(bs_dz, [2.5, 97.5])
    # permutation test: random sign flips of paired diffs
    obs = abs(md)
    cnt = 0
    for _ in range(B):
        signs = RNG.choice([-1, 1], n)
        if abs((diff * signs).mean()) >= obs:
            cnt += 1
    p_perm = (cnt + 1) / (B + 1)
    # paired t-test for reference
    t, p_t = stats.ttest_rel(xa, xb)
    dz = cohens_dz(diff)
    return dict(n=n, mean_a=xa.mean(), mean_b=xb.mean(), mean_diff=md,
               ci_md=tuple(ci_md), dz=dz, ci_dz=tuple(ci_dz),
               p_perm=p_perm, p_ttest=float(p_t),
               power=achieved_power(dz, n), n_req=required_n(dz))


def _latest_hotpotqa_report() -> tuple[str, int]:
    """Return (filename, n_questions) for the best available HotpotQA report.

    Among exp2_multihop_hotpotqa_*.json files, picks the one with the most
    questions (not the most recent timestamp) so a smoke run never shadows a
    larger result. Excludes files marked "_CONTAMINATED" (recycled DFS/Beam
    cache-bug artifacts, see Section 4.3/6.2); ties at the same n are broken
    by the timestamp embedded in the filename (falling back to file mtime),
    so a fresh clean re-run always wins over a stale one. Falls back to
    hotpotqa_results.json when no exp2 file exists.
    """
    import json as _json
    best_name, best_n, best_key = "hotpotqa_results.json", 20, ""
    for path in sorted(REP.glob("exp2_multihop_hotpotqa_*.json")):
        if "_CONTAMINATED" in path.name:
            continue
        try:
            d = _json.load(open(path, encoding="utf-8"))
            pq = d.get("per_question", {})
            first = next(iter(pq), None)
            n = len(pq[first]) if first else 0
        except Exception:
            continue
        stem = path.stem.replace("exp2_multihop_hotpotqa_", "")
        key = stem if (len(stem) == 15 and stem[8] == "_") else f"{path.stat().st_mtime:020.6f}"
        if n > best_n or (n == best_n and key > best_key):
            best_name, best_n, best_key = path.name, n, key
    return best_name, best_n


_hpqa_file, _hpqa_n = _latest_hotpotqa_report()

BENCH = {
    "General (n=204)": ("evaluation_20260623_033721.json", "treerag_dfs",
                        ["bm25", "dense", "flatrag", "raptor"]),
    "Medical (n=42)":  ("medical_results.json", "treerag_dfs",
                        ["bm25", "dense", "flatrag", "raptor"]),
    f"HotpotQA (n={_hpqa_n})": (_hpqa_file, "treerag_beam",
                                 ["bm25", "dense", "flatrag", "raptor"]),
}

if __name__ == "__main__":
    out = {}
    for name, (fname, treerag, baselines) in BENCH.items():
        pq = load_pq(fname)
        print("\n" + "=" * 78)
        print(f"{name}   TreeRAG system = {treerag}")
        print("=" * 78)
        hdr = f"{'vs baseline':<10} {'n':>3} {'ΔROUGE-L':>9} {'95% CI diff':>20} {'d_z':>6} {'95% CI d':>16} {'p_perm':>8} {'power':>6} {'n@.8':>6}"
        print(hdr)
        out[name] = {}
        for b in baselines:
            xa, xb = aligned(pq, treerag, b)
            if len(xa) == 0:
                print(f"{b:<10} (no aligned questions)"); continue
            r = analyze(xa, xb)
            out[name][b] = r
            print(f"{b:<10} {r['n']:>3} {r['mean_diff']:>+9.3f} "
                  f"[{r['ci_md'][0]:>+6.3f},{r['ci_md'][1]:>+6.3f}] "
                  f"{r['dz']:>6.3f} [{r['ci_dz'][0]:>+5.2f},{r['ci_dz'][1]:>+5.2f}] "
                  f"{r['p_perm']:>8.4f} {r['power']:>6.3f} {r['n_req']:>6}")
    json.dump(out, open(REP / "robust_stats_summary.json", "w"),
              indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print("\nSaved -> data/benchmark_reports/robust_stats_summary.json")
