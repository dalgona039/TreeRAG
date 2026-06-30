#!/usr/bin/env python3
"""Robust small-sample statistics for the FAIR (generator-controlled) protocol.

This is the protocol the ACM manuscript treats as primary: every system feeds
its retrieved context to the same local Llama 3.1 8B; only the retriever differs.
Reports assumption-free permutation tests, bootstrap CIs, effect sizes, and
a power analysis — consistent with the manuscript's honest small-sample framing.

Data sources (auto-detected — largest available file wins):
  - online_local_llama_general_v3_n100.json  (n=100, preferred)
    or online_local_llama_general_v2.json    (n=40, fallback)
  - exp1_citation_20260628_215738.json : per-question category + citation_f1
  - exp2_multihop_hotpotqa_<timestamp>.json  (largest n wins, HotpotQA fair)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data" / "benchmark_reports"
RNG = np.random.default_rng(20260630)
B = 10000


def _best_general_report() -> tuple[str, int]:
    """Return (filename, n) for the largest available fair-protocol general report."""
    preferred = "online_local_llama_general_v3_n100.json"
    fallback = "online_local_llama_general_v2.json"
    for fname in (preferred, fallback):
        path = REP / fname
        if path.is_file():
            try:
                d = json.load(open(path, encoding="utf-8"))
                pq = d.get("per_question", {})
                first = next(iter(pq), None)
                n = len(pq[first]) if first else 0
                if n > 0:
                    return fname, n
            except Exception:
                continue
    return fallback, 40


def _best_hotpotqa_report() -> tuple[str, int]:
    """Return (filename, n) for the largest available HotpotQA fair-protocol report."""
    best_name, best_n = "exp2_multihop_hotpotqa_20260629_133031.json", 0
    for path in REP.glob("exp2_multihop_hotpotqa_*.json"):
        try:
            d = json.load(open(path, encoding="utf-8"))
            pq = d.get("per_question", {})
            first = next(iter(pq), None)
            n = len(pq[first]) if first else 0
            if n > best_n:
                best_name, best_n = path.name, n
        except Exception:
            continue
    return best_name, best_n


def load_pq(fname):
    return json.load(open(REP / fname, encoding="utf-8"))["per_question"]


def categories(fname="exp1_citation_20260628_215738.json"):
    pq = load_pq(fname)
    any_sys = next(iter(pq))
    return {r["question_id"]: r.get("category") for r in pq[any_sys]}


def aligned(pq, sa, sb, metric, keep_ids=None):
    a = {r["question_id"]: r.get(metric) for r in pq[sa]}
    b = {r["question_id"]: r.get(metric) for r in pq[sb]}
    ids = [q for q in a if q in b and a[q] is not None and b[q] is not None
           and (keep_ids is None or q in keep_ids)]
    return (np.array([a[q] for q in ids], float),
            np.array([b[q] for q in ids], float))


def required_n(dz, power=0.80, alpha=0.05):
    if dz == 0:
        return float("inf")
    from scipy.stats import norm
    return int(np.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) / abs(dz)) ** 2 + 1))


def achieved_power(dz, n, alpha=0.05):
    if n < 2:
        return float("nan")
    df = n - 1
    ncp = abs(dz) * np.sqrt(n)
    tc = stats.t.ppf(1 - alpha / 2, df)
    p = float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    if not np.isfinite(p):
        p = 1.0 if ncp > tc + 5 else float("nan")
    return min(p, 1.0)


def analyze(xa, xb):
    diff = xa - xb
    n = len(diff)
    md = diff.mean()
    bs = np.array([(lambda d: d.mean())(diff[RNG.integers(0, n, n)]) for _ in range(B)])
    ci = np.percentile(bs, [2.5, 97.5])
    obs = abs(md)
    cnt = sum(abs((diff * RNG.choice([-1, 1], n)).mean()) >= obs for _ in range(B))
    p_perm = (cnt + 1) / (B + 1)
    sd = diff.std(ddof=1)
    dz = md / sd if sd > 0 else 0.0
    t, p_t = stats.ttest_rel(xa, xb)
    return dict(n=n, mean_a=xa.mean(), mean_b=xb.mean(), md=md, ci=tuple(ci),
                dz=dz, p_perm=p_perm, p_t=float(p_t),
                power=achieved_power(dz, n), n80=required_n(dz))


def row(label, r):
    return (f"{label:<26} {r['n']:>3} {r['mean_a']:>5.3f} {r['mean_b']:>5.3f} "
            f"{r['md']:>+7.3f} [{r['ci'][0]:>+6.3f},{r['ci'][1]:>+6.3f}] "
            f"{r['dz']:>+6.2f} {r['p_perm']:>7.4f} {r['p_t']:>7.4f} "
            f"{r['power']:>6.3f} {r['n80']:>6}")


HDR = (f"{'comparison':<26} {'n':>3} {'TR':>5} {'base':>5} {'Δ':>7} "
       f"{'95% CI':>16} {'d_z':>6} {'p_prm':>7} {'p_t':>7} {'pow':>6} {'n80':>6}")

if __name__ == "__main__":
    gen_fname, gen_n = _best_general_report()
    hp_fname, hp_n = _best_hotpotqa_report()
    print(f"[fair stats] General report : {gen_fname}  (n={gen_n})")
    print(f"[fair stats] HotpotQA report: {hp_fname}  (n={hp_n})")

    out = {}
    main = load_pq(gen_fname)
    cat = categories()
    mh_ids = {q for q, c in cat.items() if c == "multi_hop"}

    print(f"\n# Fair protocol — Full Benchmark (n={gen_n}), LLM-Judge: TreeRAG-Beam vs baselines")
    print(HDR)
    out["fullbench_llm_judge"] = {"_meta": {"file": gen_fname, "n": gen_n}}
    for b in ["bm25", "dense", "flatrag", "raptor", "treerag_dfs"]:
        xa, xb = aligned(main, "treerag_beam", b, "llm_judge")
        r = analyze(xa, xb); out["fullbench_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    print(f"\n# Fair protocol — Full Benchmark (n={gen_n}), ROUGE-L: TreeRAG-Beam vs baselines")
    print(HDR)
    out["fullbench_rouge"] = {"_meta": {"file": gen_fname, "n": gen_n}}
    for b in ["bm25", "dense", "flatrag", "raptor", "treerag_dfs"]:
        xa, xb = aligned(main, "treerag_beam", b, "rouge_l")
        r = analyze(xa, xb); out["fullbench_rouge"][b] = r
        print(row(f"Beam vs {b}", r))

    mh_n = sum(1 for c in cat.values() if c == "multi_hop")
    print(f"\n# Multi-hop cell (n≈{mh_n}), LLM-Judge: the manuscript's p=0.044 claim")
    print(HDR)
    out["multihop_llm_judge"] = {}
    for b in ["dense", "bm25", "flatrag", "raptor"]:
        xa, xb = aligned(main, "treerag_beam", b, "llm_judge", keep_ids=mh_ids)
        r = analyze(xa, xb); out["multihop_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    print(f"\n# Citation F1 — Full Benchmark: TreeRAG-Beam vs baselines")
    print(HDR)
    citpq = load_pq("exp1_citation_20260628_215738.json")
    out["citation_f1"] = {}
    for b in ["bm25", "dense", "flatrag", "raptor"]:
        xa, xb = aligned(citpq, "treerag_beam", b, "citation_f1")
        r = analyze(xa, xb); out["citation_f1"][b] = r
        print(row(f"Beam vs {b}", r))

    print(f"\n# HotpotQA (n={hp_n}), LLM-Judge: TreeRAG-Beam vs baselines")
    print(HDR)
    hp = load_pq(hp_fname)
    out["hotpot_llm_judge"] = {"_meta": {"file": hp_fname, "n": hp_n}}
    for b in ["bm25", "dense", "flatrag", "raptor"]:
        xa, xb = aligned(hp, "treerag_beam", b, "llm_judge")
        r = analyze(xa, xb); out["hotpot_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    json.dump(out, open(REP / "robust_stats_fair_summary.json", "w"),
              indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print("\nSaved -> data/benchmark_reports/robust_stats_fair_summary.json")
