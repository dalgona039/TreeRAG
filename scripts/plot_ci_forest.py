#!/usr/bin/env python3
"""
Generate a 95%-CI forest plot (Figure 9) summarizing the robust small-sample
statistics already reported in Table 13 of the manuscript ("Robust
small-sample statistics for PageTree-RAG versus each baseline").

Values below are transcribed verbatim from Table 13 in TreeRAG_TIST_ACM.docx
(paired mean difference Delta, 95% bootstrap CI; General/Medical rows use
PageTree-RAG (DFS), HotpotQA rows use PageTree-RAG (Beam) -- see Table 13's
caption). No new numbers are computed here; this is a visual companion to
the existing table, addressing the reviewer request for a confidence-interval
figure rather than a table-only presentation.

Writes data/benchmark_reports/figures/figure_9_ci_forest.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UNIFIED_STYLE_INJECTED (matches scripts/plot_results.py)
plt.rcParams.update({
    "axes.grid": True, "axes.grid.axis": "x", "grid.alpha": 0.28,
    "grid.linestyle": "-", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 10.5,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "font.size": 12, "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "legend.fontsize": 10, "legend.frameon": False,
    "savefig.dpi": 300, "figure.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "data" / "benchmark_reports" / "figures"

# (benchmark, metric, baseline, n, delta, ci_lo, ci_hi, p_perm_significant)
# Transcribed from Table 13. General/Medical rows: PageTree-RAG (DFS) vs.
# baseline. HotpotQA rows: PageTree-RAG (Beam) vs. baseline (Beam is the
# system actually deployed in that experiment) -- corrected for the
# cache-key and empty-context bugs described in Section 6.2.
ROWS = [
    ("General",  "ROUGE-L",   "BM25",    204, 0.128, 0.100, 0.156, True),
    ("General",  "ROUGE-L",   "Dense",   204, 0.161, 0.136, 0.187, True),
    ("General",  "ROUGE-L",   "FlatRAG", 204, 0.203, 0.175, 0.232, True),
    ("General",  "ROUGE-L",   "RAPTOR",  204, 0.257, 0.228, 0.286, True),
    ("Medical",  "ROUGE-L",   "BM25",    42,  0.007, -0.021, 0.031, False),
    ("Medical",  "ROUGE-L",   "Dense",   42,  0.051, 0.021, 0.086, True),
    ("Medical",  "ROUGE-L",   "FlatRAG", 42,  0.095, 0.071, 0.115, True),
    ("Medical",  "ROUGE-L",   "RAPTOR",  42,  0.312, 0.282, 0.342, True),
    ("HotpotQA", "ROUGE-L",   "BM25",    100, -0.130, -0.201, -0.059, True),
    ("HotpotQA", "ROUGE-L",   "Dense",   100, -0.039, -0.102, 0.020, False),
    ("HotpotQA", "ROUGE-L",   "FlatRAG", 100, -0.068, -0.134, -0.007, True),
    ("HotpotQA", "ROUGE-L",   "RAPTOR",  100, -0.017, -0.083, 0.045, False),
    ("HotpotQA", "LLM-Judge", "BM25",    100, 0.012, -0.019, 0.044, False),
    ("HotpotQA", "LLM-Judge", "Dense",   100, 0.045, 0.007, 0.081, True),
    ("HotpotQA", "LLM-Judge", "FlatRAG", 100, 0.014, -0.020, 0.051, False),
    ("HotpotQA", "LLM-Judge", "RAPTOR",  100, 0.059, 0.022, 0.098, True),
]

GROUP_COLORS = {
    "General": "#0072B2",
    "Medical": "#E69F00",
    "HotpotQA": "#009E73",
}


def main():
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y_labels = []
    y = 0
    yticks = []
    group_spans = {}
    prev_group = None
    for row in ROWS:
        bench, metric, base, n, d, lo, hi, sig = row
        group_key = f"{bench} · {metric}"
        if group_key != prev_group:
            group_spans.setdefault(group_key, [len(y_labels), None])
            if prev_group is not None:
                group_spans[prev_group][1] = len(y_labels) - 1
        prev_group = group_key
        y_labels.append(f"vs. {base}  (n={n})")
        yticks.append(y)
        y += 1
    group_spans[prev_group][1] = len(y_labels) - 1

    y_pos = list(range(len(ROWS)))[::-1]  # top row = first entry
    for idx, (row, yp) in enumerate(zip(ROWS, y_pos)):
        bench, metric, base, n, d, lo, hi, sig = row
        color = GROUP_COLORS[bench]
        marker = "o" if sig else "o"
        face = color if sig else "white"
        ax.plot([lo, hi], [yp, yp], color=color, linewidth=1.6, zorder=2)
        ax.scatter([d], [yp], s=70, color=face, edgecolors=color,
                   linewidths=1.6, zorder=3, marker=marker)

    ax.axvline(0, color="#444444", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(list(range(len(ROWS)))[::-1])
    ax.set_yticklabels([f"{r[0]} · {r[1]} vs. {r[2]} (n={r[3]})" for r in ROWS])
    ax.set_xlabel("Δ (PageTree-RAG − baseline), 95% bootstrap CI\n"
                  "(General/Medical: DFS; HotpotQA: Beam)")
    ax.set_title("Paired effect sizes with 95% confidence intervals\n"
                  "(filled = significant at $p_{perm}<0.05$; open = not significant)")

    # Group separators + labels on the right
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=c, linestyle="", markersize=8,
                       label=g) for g, c in GROUP_COLORS.items()]
    handles.append(Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
                           linestyle="", markersize=7, label="Significant ($p_{perm}<0.05$)"))
    handles.append(Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
                           linestyle="", markersize=7, label="Not significant"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.95, fontsize=9)

    lo_all = min(r[5] for r in ROWS)
    hi_all = max(r[6] for r in ROWS)
    margin = 0.05 * (hi_all - lo_all)
    ax.set_xlim(lo_all - margin, hi_all + margin)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"figure_9_ci_forest.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {FIGDIR / 'figure_9_ci_forest.png'}")


if __name__ == "__main__":
    main()
