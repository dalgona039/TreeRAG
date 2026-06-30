#!/usr/bin/env python3
"""Ablation sweep: beam width / tree depth / relevance weights / compression threshold.

OFAT design — one hyperparameter varies while the rest stay at paper defaults:
  W=5, depth=5, weights=(sem=0.6, kw=0.2, struct=0.2), compress_thresh=0.7

Runs only TreeRAG-Beam to isolate retrieval effects.
Generation and judging: llama3.1:8b via Ollama.

Usage:
  python scripts/ablation_sweep.py                     # n=50, seed=42
  python scripts/ablation_sweep.py --n 40 --seed 0
  python scripts/ablation_sweep.py --dry-run           # print config only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORT_DIR = _ROOT / "data" / "benchmark_reports"
DATASET_PATH = _ROOT / "benchmarks" / "datasets" / "full_benchmark.json"

# ── Paper defaults ────────────────────────────────────────────────────────────
DEFAULT_BEAM_WIDTH = 5
DEFAULT_MAX_DEPTH = 5
DEFAULT_SEM_W = 0.6
DEFAULT_KW_W = 0.2
DEFAULT_STRUCT_W = 0.2
DEFAULT_COMPRESS_THRESH = 0.7

N_QUESTIONS = 50
SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(vals: List[Any]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def load_questions(n: int, seed: int) -> List[Dict[str, Any]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    rng = random.Random(seed)
    rng.shuffle(qs)
    return qs[:n]


def run_beam_config(
    questions: List[Dict[str, Any]],
    evaluator,
    beam_width: int,
    max_depth: int,
    sem_w: float,
    kw_w: float,
    struct_w: float,
    compress_thresh: float,
) -> List[Dict[str, Any]]:
    """Run TreeRAG-Beam on all questions with given hyperparams.

    Creates a fresh TreeRAGReasoner per question (no cross-config cache leakage).
    Monkey-patches BeamSearchNavigator class weights for this call, then restores.
    """
    from src.core.beam_search import BeamSearchNavigator
    from src.core.reasoner import TreeRAGReasoner

    # Save and patch class-level beam weights
    _orig = (
        BeamSearchNavigator.SEMANTIC_WEIGHT,
        BeamSearchNavigator.KEYWORD_WEIGHT,
        BeamSearchNavigator.STRUCTURE_WEIGHT,
    )
    BeamSearchNavigator.SEMANTIC_WEIGHT = sem_w
    BeamSearchNavigator.KEYWORD_WEIGHT = kw_w
    BeamSearchNavigator.STRUCTURE_WEIGHT = struct_w

    rows: List[Dict[str, Any]] = []
    try:
        for q in questions:
            t0 = time.perf_counter()
            try:
                reasoner = TreeRAGReasoner(
                    [q["document_id"]],
                    traversal_algorithm="beam_search",
                    beam_width=beam_width,
                    enable_compression=True,
                )
                if reasoner.compressor is not None:
                    reasoner.compressor.similarity_threshold = compress_thresh
                answer, meta = reasoner.query(
                    q["question"],
                    max_depth=max_depth,
                    use_simple_prompt=True,
                )
                nodes = meta.get("nodes_selected", []) or []
            except Exception as exc:
                print(f"  ⚠ {q['question_id']}: {exc}")
                answer, nodes = "", []
            latency = time.perf_counter() - t0

            ctx = " ".join(
                n.get("summary", n.get("title", "")) if isinstance(n, dict) else str(n)
                for n in nodes
            )
            expected = q.get("expected_answer_hint", "")
            scores = evaluator.score_answer(q["question"], ctx, answer, expected)
            rows.append({
                "question_id": q["question_id"],
                "latency": latency,
                "context_tokens": int(len(ctx) / 4),
                **scores,
            })
    finally:
        # Always restore class weights
        BeamSearchNavigator.SEMANTIC_WEIGHT = _orig[0]
        BeamSearchNavigator.KEYWORD_WEIGHT = _orig[1]
        BeamSearchNavigator.STRUCTURE_WEIGHT = _orig[2]

    return rows


def agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "rouge_l": _mean([r.get("rouge_l") for r in rows]),
        "llm_judge": _mean([r.get("llm_judge") for r in rows]),
        "context_tokens": _mean([r.get("context_tokens") for r in rows]),
        "latency_s": _mean([r.get("latency") for r in rows]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TreeRAG ablation sweep (OFAT)")
    parser.add_argument("--n", type=int, default=N_QUESTIONS,
                        help="Questions per config (default=50)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=str(REPORT_DIR / "ablation_sweep_llama.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs and exit without running")
    args = parser.parse_args()

    from src.core.ollama_client import OllamaClient
    from src.config import set_client_override
    from benchmarks.run_real_evaluation import Evaluator

    client = OllamaClient(model="llama3.1:8b")
    set_client_override(client)
    evaluator = Evaluator(
        mode="online",
        use_llm_judge=True,
        local_judge=True,
        local_judge_model="llama3.1:8b",
        gen_backend="ollama",
        gen_model="llama3.1:8b",
    )

    questions = load_questions(args.n, args.seed)
    print(f"Loaded {len(questions)} questions  (seed={args.seed})")

    results: List[Dict[str, Any]] = []

    def run_sweep(sweep_name: str, value, label: str, is_default: bool, **overrides):
        cfg = dict(
            beam_width=DEFAULT_BEAM_WIDTH,
            max_depth=DEFAULT_MAX_DEPTH,
            sem_w=DEFAULT_SEM_W,
            kw_w=DEFAULT_KW_W,
            struct_w=DEFAULT_STRUCT_W,
            compress_thresh=DEFAULT_COMPRESS_THRESH,
        )
        cfg.update(overrides)
        if args.dry_run:
            print(f"  [{sweep_name}={value}]  " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
            results.append({"sweep": sweep_name, "value": value, "label": label,
                            "is_default": is_default, "dry_run": True})
            return
        print(f"\n▶ {sweep_name} = {value}  ({label})")
        rows = run_beam_config(questions, evaluator, **cfg)
        r = agg(rows)
        entry = {"sweep": sweep_name, "value": value, "label": label,
                 "is_default": is_default, **r}
        results.append(entry)
        print(f"   ROUGE-L={r['rouge_l']}  LLM-Judge={r['llm_judge']}  "
              f"ctx_tok={r['context_tokens']}  lat={r['latency_s']}s")

    # ── Sweep 1: Beam width ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sweep 1: Beam Width W ∈ {1, 3, 5*, 8}")
    print("=" * 60)
    for w in [1, 3, 5, 8]:
        run_sweep("beam_width", w, f"W={w}", w == DEFAULT_BEAM_WIDTH, beam_width=w)

    # ── Sweep 2: Tree depth ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sweep 2: Max Depth D ∈ {2, 3, 5*, 10}")
    print("=" * 60)
    for d in [2, 3, 5, 10]:
        run_sweep("max_depth", d, f"D={d}", d == DEFAULT_MAX_DEPTH, max_depth=d)

    # ── Sweep 3: Relevance weights ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sweep 3: Weights (sem/kw/struct) ∈ {0.6/0.2/0.2*, 0.5/0.3/0.2, 0.8/0.1/0.1, 1.0/0.0/0.0}")
    print("=" * 60)
    weight_cfgs: List[Tuple[str, float, float, float]] = [
        ("0.6/0.2/0.2", 0.6, 0.2, 0.2),
        ("0.5/0.3/0.2", 0.5, 0.3, 0.2),
        ("0.8/0.1/0.1", 0.8, 0.1, 0.1),
        ("1.0/0.0/0.0", 1.0, 0.0, 0.0),
    ]
    for label, s, k, st in weight_cfgs:
        is_def = (s, k, st) == (DEFAULT_SEM_W, DEFAULT_KW_W, DEFAULT_STRUCT_W)
        run_sweep("weights", f"{s}/{k}/{st}", label, is_def, sem_w=s, kw_w=k, struct_w=st)

    # ── Sweep 4: Compression threshold ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sweep 4: Compression Threshold τ ∈ {0.5, 0.6, 0.7*, 0.8}")
    print("=" * 60)
    for t in [0.5, 0.6, 0.7, 0.8]:
        run_sweep("compress_thresh", t, f"τ={t}", t == DEFAULT_COMPRESS_THRESH,
                  compress_thresh=t)

    # ── Save & print table ───────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"n_questions": args.n, "seed": args.seed, "rows": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved → {args.output}")

    print("\n" + "=" * 82)
    print(f"{'Sweep':<16} {'Value':<16} {'ROUGE-L':>8} {'LLM-Judge':>10} "
          f"{'ctx_tok':>8} {'lat(s)':>7} {'*dflt':>5}")
    print("-" * 82)
    for r in results:
        dflt = "✓" if r.get("is_default") else ""
        rl = f"{r['rouge_l']:.4f}" if r.get("rouge_l") is not None else "   —  "
        lj = f"{r['llm_judge']:.4f}" if r.get("llm_judge") is not None else "   —  "
        ct = f"{r['context_tokens']:.0f}" if r.get("context_tokens") is not None else " —"
        lt = f"{r['latency_s']:.1f}" if r.get("latency_s") is not None else " —"
        print(f"{r['sweep']:<16} {str(r['value']):<16} {rl:>8} {lj:>10} "
              f"{ct:>8} {lt:>7} {dflt:>5}")


if __name__ == "__main__":
    main()
