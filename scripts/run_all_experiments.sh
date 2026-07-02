#!/usr/bin/env bash
# run_all_experiments.sh — 리뷰어 보강 실험 전 자동 실행 스크립트
# P0-A/B가 이미 실행 중이면 종료를 기다린 뒤 P1-C → P1-D → 통계 → 그래프 → 검증 순서로 실행.
# Usage: bash scripts/run_all_experiments.sh [P0A_PID] [P0B_PID]

set -euo pipefail
LOG_DIR="/private/tmp/claude-501/-Volumes-a3122a1-TreeRAG/152f97d7-7421-46c3-8188-5b38fdb14e10/scratchpad"
VENV=".venv/bin/python"
REP="data/benchmark_reports"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }

# ── 실행 중인 P0-A / P0-B PID 기다리기 ─────────────────────────────────────
P0A_PID="${1:-}"
P0B_PID="${2:-}"

if [ -n "$P0A_PID" ]; then
  log "P0-A (PID $P0A_PID) 완료 대기 중..."
  while kill -0 "$P0A_PID" 2>/dev/null; do sleep 10; done
  log "P0-A 완료"
fi

if [ -n "$P0B_PID" ]; then
  log "P0-B (PID $P0B_PID) 완료 대기 중..."
  while kill -0 "$P0B_PID" 2>/dev/null; do sleep 10; done
  log "P0-B 완료"
fi

# ── P1-C: 독립 Judge 교차검증 (gen=llama3.1:8b, judge=qwen3.5:9b) ──────────
log "=== P1-C: 독립 judge 교차검증 (qwen3.5:9b) 시작 ==="
$VENV benchmarks/run_real_evaluation.py \
  --dataset benchmarks/datasets/full_benchmark.json \
  --systems all --limit 100 --seed 42 \
  --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model qwen3.5:9b \
  --output "$REP/online_local_qwen_judge_n100.json" \
  2>&1 | tee "$LOG_DIR/p1c_qwen_judge.log"
log "P1-C 완료 → $REP/online_local_qwen_judge_n100.json"

# ── P1-D: GovReport 긴 문서 벤치마크 ───────────────────────────────────────
log "=== P1-D: GovReport 40문항 평가 시작 ==="
$VENV benchmarks/run_real_evaluation.py \
  --dataset benchmarks/datasets/govreport_benchmark.json \
  --systems all --limit 40 --seed 42 \
  --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output "$REP/online_local_llama_govreport.json" \
  2>&1 | tee "$LOG_DIR/p1d_govreport.log"
log "P1-D 완료 → $REP/online_local_llama_govreport.json"

# ── 강건 통계 재계산 ─────────────────────────────────────────────────────────
log "=== 강건 통계 재계산 ==="
$VENV scripts/robust_stats.py 2>&1 | tee "$LOG_DIR/robust_stats.log"
$VENV scripts/robust_stats_fair.py 2>&1 | tee "$LOG_DIR/robust_stats_fair.log"
log "강건 통계 완료"

# ── P1-C 결과: Spearman ρ 계산 ───────────────────────────────────────────────
log "=== P1-C: Spearman 순위 상관 계산 ==="
$VENV - <<'PY' 2>&1 | tee "$LOG_DIR/p1c_spearman.log"
import json, sys
from pathlib import Path
from scipy.stats import spearmanr

REP = Path("data/benchmark_reports")
SYSTEMS = ["bm25", "dense", "flatrag", "raptor", "treerag_dfs", "treerag_beam"]

def mean_judge(fname, sys_name):
    d = json.load(open(REP / fname, encoding="utf-8"))
    rows = d["per_question"].get(sys_name, [])
    vals = [r["llm_judge"] for r in rows if r.get("llm_judge") is not None]
    return sum(vals)/len(vals) if vals else None

llama_f = "online_local_llama_general_v3_n100.json"
qwen_f  = "online_local_qwen_judge_n100.json"

for f in [llama_f, qwen_f]:
    if not (REP / f).is_file():
        print(f"MISSING: {f}"); sys.exit(0)

llama_scores = [mean_judge(llama_f, s) for s in SYSTEMS]
qwen_scores  = [mean_judge(qwen_f,  s) for s in SYSTEMS]

pairs = [(l,q) for l,q in zip(llama_scores, qwen_scores) if l is not None and q is not None]
xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
rho, pval = spearmanr(xs, ys)

print(f"\n{'System':<18} {'LLM-Judge(llama)':>18} {'LLM-Judge(qwen)':>16}")
print("-" * 54)
for s, l, q in zip(SYSTEMS, llama_scores, qwen_scores):
    lstr = f"{l:.4f}" if l is not None else "  —  "
    qstr = f"{q:.4f}" if q is not None else "  —  "
    print(f"{s:<18} {lstr:>18} {qstr:>16}")

print(f"\nSpearman ρ = {rho:.4f}  (p = {pval:.4f})")
out = {"spearman_rho": rho, "p_value": pval,
       "llama_judge_file": llama_f, "qwen_judge_file": qwen_f,
       "system_scores": {s: {"llama": l, "qwen": q}
                         for s, l, q in zip(SYSTEMS, llama_scores, qwen_scores)}}
json.dump(out, open(REP / "p1c_judge_crossval.json", "w"), indent=2)
print(f"\nSaved → data/benchmark_reports/p1c_judge_crossval.json")
PY

# ── 그래프 재생성 ─────────────────────────────────────────────────────────────
log "=== 그래프 재생성 ==="
$VENV - <<'PY' 2>&1 | tee "$LOG_DIR/plots.log"
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from scripts.plot_results import (
    figure_architecture, figure_main_bars, figure_multihop,
    figure_context_reduction, FIG_DIR
)
import matplotlib
matplotlib.use("Agg")

REP = Path("data/benchmark_reports")

# 메인 general 리포트 (n=100 우선)
for fname in ("online_local_llama_general_v3_n100.json",
              "online_local_llama_general_v2.json"):
    if (REP / fname).is_file():
        main_report = json.load(open(REP / fname, encoding="utf-8"))
        print(f"Main report: {fname}")
        break

# HotpotQA 리포트 (가장 큰 것)
best_hp, best_n = None, 0
for p in REP.glob("exp2_multihop_hotpotqa_*.json"):
    d = json.load(open(p)); pq = d.get("per_question", {})
    first = next(iter(pq), None)
    n = len(pq[first]) if first else 0
    if n > best_n:
        best_hp, best_n = p, n
hotpot_report = json.load(open(best_hp)) if best_hp else None
print(f"HotpotQA report: {best_hp.name if best_hp else 'none'}  (n={best_n})")

FIG_DIR.mkdir(parents=True, exist_ok=True)
figure_architecture()
figure_main_bars(main_report)
if hotpot_report:
    figure_multihop(main_report, hotpot_report)
figure_context_reduction(main_report)
print(f"Figures saved to {FIG_DIR}")
PY

# ── 최종 검증 ────────────────────────────────────────────────────────────────
log "=== 최종 검증 ==="
$VENV - <<'PY' 2>&1 | tee "$LOG_DIR/final_check.log"
import json
from pathlib import Path

REP = Path("data/benchmark_reports")
checks = {
    "P0-A n=100 general": "online_local_llama_general_v3_n100.json",
    "P0-B ablation sweep": "ablation_sweep_llama.json",
    "P1-C qwen judge":     "online_local_qwen_judge_n100.json",
    "P1-C spearman":       "p1c_judge_crossval.json",
    "P1-D govreport":      "online_local_llama_govreport.json",
    "robust_stats":        "robust_stats_summary.json",
    "robust_stats_fair":   "robust_stats_fair_summary.json",
}

all_ok = True
print(f"\n{'Artifact':<30} {'Status':>10} {'Detail'}")
print("-" * 70)
for label, fname in checks.items():
    p = REP / fname
    if p.is_file():
        d = json.load(open(p, encoding="utf-8"))
        pq = d.get("per_question", {})
        first = next(iter(pq), None)
        n = len(pq[first]) if first else "—"
        print(f"{label:<30} {'✓':>10}  n={n}")
    else:
        print(f"{label:<30} {'MISSING':>10}  {fname}")
        all_ok = False

if all_ok:
    print("\n✅ 모든 산출물 확인 완료")
else:
    print("\n⚠ 일부 파일 누락 — 위 목록 확인")
PY

log "=== 전체 완료 ==="
