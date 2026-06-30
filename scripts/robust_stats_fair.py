#!/usr/bin/env python3
"""Robust small-sample statistics for the FAIR (generator-controlled) protocol.

This is the protocol the ACM manuscript treats as primary: every system feeds
its retrieved context to the same local Llama 3.1 8B; only the retriever differs.
Sample sizes are small (Full Benchmark n=40, HotpotQA n=20, multi-hop cell n=11),
so we report assumption-free permutation tests, bootstrap CIs, effect sizes, and
a power analysis — consistent with the manuscript's honest small-sample framing.

Data sources:
  - online_local_llama_general_v2.json : n=40 fair-protocol main results
    (per-question rouge_l, bertscore, llm_judge, context_tokens)
  - exp1_citation_20260628_215738.json : n=40, per-question category + citation_f1
  - exp2_multihop_hotpotqa_20260629_133031.json : n=20 HotpotQA fair protocol
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
    out = {}
    main = load_pq("online_local_llama_general_v2.json")
    cat = categories()
    mh_ids = {q for q, c in cat.items() if c == "multi_hop"}

    print("\n# Fair protocol — Full Benchmark (n=40), LLM-Judge: TreeRAG-Beam vs baselines")
    print(HDR)
    out["fullbench_llm_judge"] = {}
    for b in ["bm25", "dense", "flatrag", "raptor", "treerag_dfs"]:
        xa, xb = aligned(main, "treerag_beam", b, "llm_judge")
        r = analyze(xa, xb); out["fullbench_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    print("\n# Fair protocol — Full Benchmark (n=40), ROUGE-L: TreeRAG-Beam vs baselines")
    print(HDR)
    out["fullbench_rouge"] = {}
    for b in ["bm25", "dense", "flatrag", "raptor", "treerag_dfs"]:
        xa, xb = aligned(main, "treerag_beam", b, "rouge_l")
        r = analyze(xa, xb); out["fullbench_rouge"][b] = r
        print(row(f"Beam vs {b}", r))

    print("\n# Multi-hop cell (n=11), LLM-Judge: the manuscript's p=0.044 claim")
    print(HDR)
    out["multihop_llm_judge"] = {}
    for b in ["dense", "bm25", "flatrag", "raptor"]:
        xa, xb = aligned(main, "treerag_beam", b, "llm_judge", keep_ids=mh_ids)
        r = analyze(xa, xb); out["multihop_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    print("\n# Citation F1 — Full Benchmark (n=40): TreeRAG-Beam vs baselines")
    print(HDR)
    citpq = load_pq("exp1_citation_20260628_215738.json")
    out["citation_f1"] = {}
    for b in ["bm25", "dense", "flatrag", "raptor"]:
        xa, xb = aligned(citpq, "treerag_beam", b, "citation_f1")
        r = analyze(xa, xb); out["citation_f1"][b] = r
        print(row(f"Beam vs {b}", r))

    print("\n# HotpotQA (n=20), LLM-Judge: TreeRAG-Beam vs baselines")
    print(HDR)
    hp = load_pq("exp2_multihop_hotpotqa_20260629_133031.json")
    out["hotpot_llm_judge"] = {}
    for b in ["bm25", "dense", "flatrag", "raptor"]:
        xa, xb = aligned(hp, "treerag_beam", b, "llm_judge")
        r = analyze(xa, xb); out["hotpot_llm_judge"][b] = r
        print(row(f"Beam vs {b}", r))

    json.dump(out, open(REP / "robust_stats_fair_summary.json", "w"),
              indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print("\nSaved -> data/benchmark_reports/robust_stats_fair_summary.json")
