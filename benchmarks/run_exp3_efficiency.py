#!/usr/bin/env python3
"""
Experiment 3: Context Efficiency

Loads an existing benchmark JSON report (produced by run_real_evaluation.py)
and computes per-system efficiency metrics:

  avg_context_tokens  –  mean tokens fed to the generator per question
  avg_llm_judge       –  mean LLM-judge score
  avg_rouge_l         –  mean ROUGE-L score
  avg_latency         –  mean generation latency (s)

Writes exp3_efficiency_<date>.{json,md} to data/benchmark_reports/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = _ROOT / "data" / "benchmark_reports"

SYSTEM_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "flatrag": "FlatRAG",
    "raptor": "RAPTOR",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}


def _mean(vals: List[Any]) -> float:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_efficiency(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    per_q = report.get("per_question", {})
    result = {}
    for system, rows in per_q.items():
        result[system] = {
            "avg_context_tokens": _mean([r.get("context_tokens") for r in rows]),
            "avg_llm_judge": _mean([r.get("llm_judge") for r in rows]),
            "avg_rouge_l": _mean([r.get("rouge_l") for r in rows]),
            "avg_latency": _mean([r.get("latency") for r in rows]),
            "n": len(rows),
        }
    return result


def print_table(eff: Dict[str, Dict[str, float]], systems: List[str]) -> None:
    print()
    print("┌─────────────────┬──────────────┬───────────┬─────────┬──────────┐")
    print("│ System          │ CTX Tokens   │ LLM-Judge │ ROUGE-L │ Latency  │")
    print("├─────────────────┼──────────────┼───────────┼─────────┼──────────┤")
    for s in systems:
        e = eff.get(s, {})
        ct = e.get("avg_context_tokens", 0)
        lj = e.get("avg_llm_judge")
        rl = e.get("avg_rouge_l", 0)
        lt = e.get("avg_latency", 0)
        lj_str = f"{lj:.3f}" if lj is not None else "  —  "
        print(
            f"│ {SYSTEM_LABELS.get(s, s):<15} │ {ct:>8.0f}     │"
            f"   {lj_str:^5} │  {rl:.3f}  │  {lt:.2f}s │"
        )
    print("└─────────────────┴──────────────┴───────────┴─────────┴──────────┘")
    print()
    # Pareto note: highest judge / lowest tokens
    by_judge = sorted(
        [(s, eff[s]["avg_llm_judge"] or 0, eff[s]["avg_context_tokens"]) for s in systems if s in eff],
        key=lambda x: -x[1],
    )
    print("  Pareto (LLM-Judge ↑, CTX Tokens ↓) ranking:")
    for rank, (s, lj, ct) in enumerate(by_judge, 1):
        print(f"    {rank}. {SYSTEM_LABELS.get(s,s):18s}  judge={lj:.3f}  tokens={ct:.0f}")


def save_markdown(
    eff: Dict[str, Dict[str, float]], systems: List[str], path: Path, source_file: str
) -> None:
    lines = [
        "# Experiment 3: Context Efficiency",
        "",
        f"Source report: `{source_file}`",
        "",
        "| System | CTX Tokens | LLM-Judge | ROUGE-L | Latency(s) |",
        "|--------|-----------|-----------|---------|-----------|",
    ]
    for s in systems:
        e = eff.get(s, {})
        lj = e.get("avg_llm_judge")
        lj_str = f"{lj:.3f}" if lj is not None else "—"
        lines.append(
            f"| {SYSTEM_LABELS.get(s,s)} "
            f"| {e.get('avg_context_tokens',0):.0f} "
            f"| {lj_str} "
            f"| {e.get('avg_rouge_l',0):.3f} "
            f"| {e.get('avg_latency',0):.3f} |"
        )
    lines += ["", "*(Lower CTX Tokens + Higher LLM-Judge = better Pareto position)*"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📄 Markdown → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Context Efficiency analysis")
    parser.add_argument(
        "--report",
        default=str(REPORT_DIR / "online_local_llama_general_v2.json"),
        help="Path to existing run_real_evaluation JSON report",
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    systems = report.get("systems", list(report.get("per_question", {}).keys()))
    n_q = report.get("summary", {}).get(systems[0], {}).get("n", "?") if systems else "?"
    print(f"Experiment 3: Context Efficiency")
    print(f"Source: {args.report}")
    print(f"Systems: {', '.join(systems)}  |  Q per system: {n_q}")

    eff = compute_efficiency(report)
    print_table(eff, systems)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"exp3_efficiency_{stamp}.json"
    md_path = REPORT_DIR / f"exp3_efficiency_{stamp}.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source": args.report, "systems": systems, "efficiency": eff},
            f, ensure_ascii=False, indent=2,
        )
    print(f"💾 JSON → {json_path}")
    save_markdown(eff, systems, md_path, Path(args.report).name)


if __name__ == "__main__":
    main()
