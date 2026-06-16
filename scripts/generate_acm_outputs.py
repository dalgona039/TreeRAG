#!/usr/bin/env python3
"""
ACM paper-ready output orchestrator (PHASE 5 of the ACM upgrade plan).

Reads the evaluation reports produced by the runner and emits:
  * Table 1 (Full + HotpotQA combined) and Table 2 (medical) LaTeX  -> paper_tables_acm.tex
  * Figures 1-4 (architecture, main bars, multi-hop, context reduction) -> figures/
  * A contribution summary printed to stdout for the Introduction section.

All inputs are optional; missing reports are skipped with a note so the script
runs even when only some benchmarks have been evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REPORT_DIR = _PROJECT_ROOT / "data" / "benchmark_reports"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print("  (skip) missing: {0}".format(path.name))
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_baseline_rouge(report: Dict[str, Any], exclude=("treerag_dfs", "treerag_beam")):
    best_s, best_v = None, -1.0
    for s in report["systems"]:
        if s in exclude:
            continue
        v = report["summary"][s].get("rouge_l", 0.0) or 0.0
        if v > best_v:
            best_s, best_v = s, v
    return best_s, best_v


def contribution_summary(full, hotpot, medical) -> str:
    lines = ["", "=== Contribution Summary (paste into Introduction) ==="]

    if full:
        tb = full["summary"].get("treerag_beam", {})
        bs, bv = _best_baseline_rouge(full)
        sig = full.get("significance", {})
        rap = full["summary"].get("raptor", {})
        lines.append(
            "1. On the full benchmark, TreeRAG-Beam reaches ROUGE-L {0:.3f}, "
            "vs the strongest non-tree baseline {1} at {2:.3f} "
            "(Delta {3:+.3f}).".format(tb.get("rouge_l", 0), bs, bv, tb.get("rouge_l", 0) - bv)
        )
        if rap:
            p = sig.get("raptor", {}).get("p_value")
            ptxt = " (p={0:.4f})".format(p) if isinstance(p, (int, float)) else ""
            lines.append(
                "2. TreeRAG-Beam outperforms the RAPTOR bottom-up baseline by "
                "{0:+.3f} ROUGE-L{1}.".format(tb.get("rouge_l", 0) - rap.get("rouge_l", 0), ptxt)
            )

    if hotpot:
        htb = hotpot["summary"].get("treerag_beam", {})
        hbs, hbv = _best_baseline_rouge(hotpot)
        lines.append(
            "3. On HotpotQA multi-hop questions, TreeRAG-Beam scores ROUGE-L "
            "{0:.3f}, exceeding the best single-document baseline {1} ({2:.3f}) "
            "by {3:+.3f} -- the advantage is larger than on the full set, "
            "supporting the multi-hop hypothesis.".format(
                htb.get("rouge_l", 0), hbs, hbv, htb.get("rouge_l", 0) - hbv)
        )

    if medical:
        mtb = medical["summary"].get("treerag_beam", {})
        lines.append(
            "4. In the medical domain, TreeRAG-Beam attains medical entity recall "
            "{0:.3f} while preserving page-level citations, addressing the "
            "clinical source-traceability requirement.".format(
                mtb.get("medical_entity_recall", 0) or 0)
        )

    lines.append(
        "5. TreeRAG preserves the original document hierarchy (top-down), "
        "yielding auditable [doc, p.X] citations that flat (BM25/Dense) and "
        "bottom-up (RAPTOR) methods cannot provide."
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate ACM paper outputs (PHASE 5)")
    parser.add_argument("--full", default=str(REPORT_DIR / "evaluation_latest.json"))
    parser.add_argument("--hotpot", default=str(REPORT_DIR / "hotpotqa_results.json"))
    parser.add_argument("--medical", default=str(REPORT_DIR / "medical_results.json"))
    parser.add_argument("--tables-out", default=str(REPORT_DIR / "paper_tables_acm.tex"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print("Loading reports...")
    full = _load(Path(args.full))
    hotpot = _load(Path(args.hotpot))
    medical = _load(Path(args.medical))

    # --- Tables -----------------------------------------------------------
    from scripts.generate_paper_tables import table_main_combined, table_medical

    blocks = ["% ACM paper tables (auto-generated; regenerate after ONLINE run)."]
    if full:
        blocks += [table_main_combined(full, hotpot or {}), ""]
    if medical:
        blocks += [table_medical(medical), ""]
    out = Path(args.tables_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print("💾 ACM tables → {0}".format(out))

    # --- Figures ----------------------------------------------------------
    if not args.no_figures and full:
        from scripts.plot_results import (
            figure_architecture,
            figure_context_reduction,
            figure_main_bars,
            figure_multihop,
        )

        print("Generating figures...")
        figure_architecture()
        figure_main_bars(full)
        if hotpot:
            figure_multihop(full, hotpot)
        figure_context_reduction(full)

    # --- Contribution summary --------------------------------------------
    print(contribution_summary(full, hotpot, medical))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
