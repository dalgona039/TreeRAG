
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


@dataclass
class PlotConfig:
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 300
    color_palette: str = "Set2"
    primary_color: str = "#2ecc71"
    secondary_color: str = "#3498db"
    accent_color: str = "#e74c3c"
    font_family: str = "serif"
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    style: str = "whitegrid"
    despine: bool = True
    output_format: str = "pdf"


class ResultPlotter:
    
    def __init__(
        self,
        output_dir: str = "results/figures",
        config: Optional[PlotConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PlotConfig()
        
        self._setup_style()
    
    def _setup_style(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        if SEABORN_AVAILABLE:
            sns.set_style(self.config.style)
            sns.set_palette(self.config.color_palette)
        plt.rcParams.update({
            "font.family": self.config.font_family,
            "font.size": self.config.tick_fontsize,
            "axes.titlesize": self.config.title_fontsize,
            "axes.labelsize": self.config.label_fontsize,
            "xtick.labelsize": self.config.tick_fontsize,
            "ytick.labelsize": self.config.tick_fontsize,
            "legend.fontsize": self.config.legend_fontsize,
            "figure.dpi": self.config.dpi,
        })
    
    def plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Performance Comparison",
        filename: str = "performance_comparison"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        systems = list(results.keys())
        n_systems = len(systems)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = 0.8 / n_systems
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_systems))
        
        for i, system in enumerate(systems):
            values = [results[system].get(m, 0) for m in metrics]
            offset = (i - n_systems/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=system, color=colors[i])
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([self._format_metric_name(m) for m in metrics])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        
        if self.config.despine and SEABORN_AVAILABLE:
            sns.despine()
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_radar_chart(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Multi-Metric Comparison",
        filename: str = "radar_chart"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        for (system, values), color in zip(results.items(), colors):
            vals = [values.get(m, 0) for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=system, color=color)
            ax.fill(angles, vals, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._format_metric_name(m) for m in metrics])
        
        ax.set_ylim(0, 1)
        ax.set_title(title, size=self.config.title_fontsize, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_efficiency_scatter(
        self,
        results: Dict[str, Dict[str, float]],
        x_metric: str = "latency_mean_ms",
        y_metric: str = "precision_at_5",
        title: str = "Efficiency vs Quality Trade-off",
        filename: str = "efficiency_scatter"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        systems = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(systems)))
        
        for system, color in zip(systems, colors):
            x = results[system].get(x_metric, 0)
            y = results[system].get(y_metric, 0)
            
            ax.scatter(x, y, s=150, color=color, label=system, edgecolors='black', linewidths=1)
            ax.annotate(system, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel(self._format_metric_name(x_metric))
        ax.set_ylabel(self._format_metric_name(y_metric))
        ax.set_title(title)
        ax.annotate('Better →', xy=(0.02, 0.98), xycoords='axes fraction',
                   ha='left', va='top', fontsize=10, style='italic',
                   color='gray')
        ax.annotate('↓ Faster', xy=(0.02, 0.02), xycoords='axes fraction',
                   ha='left', va='bottom', fontsize=10, style='italic',
                   color='gray', rotation=90)
        
        if self.config.despine and SEABORN_AVAILABLE:
            sns.despine()
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_calibration_diagram(
        self,
        bins: List[Dict[str, float]],
        title: str = "Reliability Diagram",
        filename: str = "calibration_diagram"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        confidences = [b["mean_confidence"] for b in bins if b.get("n_samples", 0) > 0]
        accuracies = [b["accuracy"] for b in bins if b.get("n_samples", 0) > 0]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.bar(confidences, accuracies, width=0.1, alpha=0.7, 
               edgecolor='black', label='Model')
        for conf, acc in zip(confidences, accuracies):
            if conf > acc:
                ax.fill_between([conf-0.05, conf+0.05], [acc, acc], [conf, conf],
                               alpha=0.3, color='red', label='_nolegend_')
            else:
                ax.fill_between([conf-0.05, conf+0.05], [conf, conf], [acc, acc],
                               alpha=0.3, color='green', label='_nolegend_')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_learning_curve(
        self,
        training_history: Dict[str, List[float]],
        title: str = "Learning Curve",
        filename: str = "learning_curve"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(training_history)))
        
        for (metric, values), color in zip(training_history.items(), colors):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, '-o', label=metric, color=color, markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / Metric')
        ax.set_title(title)
        ax.legend()
        
        if self.config.despine and SEABORN_AVAILABLE:
            sns.despine()
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_ablation_waterfall(
        self,
        ablations: Dict[str, float],
        baseline_name: str,
        metric_name: str = "Performance",
        title: str = "Ablation Study",
        filename: str = "ablation_waterfall"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = list(ablations.keys())
        deltas = list(ablations.values())
        y_starts = [0]
        for i, delta in enumerate(deltas[:-1]):
            y_starts.append(y_starts[-1] + delta)
        
        colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]
        
        bars = ax.bar(range(len(components)), deltas, bottom=y_starts,
                     color=colors, edgecolor='black', linewidth=1)
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            height = bar.get_height()
            y_pos = y_starts[i] + height / 2
            ax.annotate(f'{delta:+.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white')
        
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_ylabel(f'Δ {metric_name}')
        ax.set_title(title)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        positive_patch = mpatches.Patch(color='#2ecc71', label='Positive Impact')
        negative_patch = mpatches.Patch(color='#e74c3c', label='Negative Impact')
        ax.legend(handles=[positive_patch, negative_patch], loc='upper right')
        
        if self.config.despine and SEABORN_AVAILABLE:
            sns.despine()
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_token_distribution(
        self,
        tree_rag_tokens: List[int],
        flat_rag_tokens: List[int],
        title: str = "Token Usage Distribution",
        filename: str = "token_distribution"
    ) -> Optional[Path]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        bins = np.linspace(0, max(max(tree_rag_tokens), max(flat_rag_tokens)), 30)
        
        ax.hist(flat_rag_tokens, bins=bins, alpha=0.7, label='Flat RAG',
               color=self.config.secondary_color)
        ax.hist(tree_rag_tokens, bins=bins, alpha=0.7, label='TreeRAG',
               color=self.config.primary_color)
        ax.axvline(np.mean(flat_rag_tokens), color=self.config.secondary_color,
                  linestyle='--', linewidth=2, label=f'Flat RAG Mean: {np.mean(flat_rag_tokens):.0f}')
        ax.axvline(np.mean(tree_rag_tokens), color=self.config.primary_color,
                  linestyle='--', linewidth=2, label=f'TreeRAG Mean: {np.mean(tree_rag_tokens):.0f}')
        
        ax.set_xlabel('Token Count')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        
        if self.config.despine and SEABORN_AVAILABLE:
            sns.despine()
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.config.output_format}"
        plt.savefig(filepath, format=self.config.output_format, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _format_metric_name(self, metric: str) -> str:
        replacements = {
            "precision_at_5": "P@5",
            "recall_at_5": "R@5",
            "f1_at_5": "F1@5",
            "ndcg_at_5": "NDCG@5",
            "mrr": "MRR",
            "map": "MAP",
            "groundedness": "Groundedness",
            "hallucination_rate": "Hallucination Rate",
            "latency_mean_ms": "Latency (ms)",
            "latency_p95_ms": "P95 Latency (ms)",
            "tokens_used": "Tokens Used",
        }
        return replacements.get(metric, metric.replace("_", " ").title())


def generate_all_plots(
    results_file: str,
    output_dir: str = "results/figures"
) -> Dict[str, Path]:
    if not MATPLOTLIB_AVAILABLE:
        return {}
    
    with open(results_file) as f:
        results = json.load(f)
    
    plotter = ResultPlotter(output_dir)
    saved_plots = {}
    
    if "main_results" in results:
        metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "groundedness"]
        path = plotter.plot_performance_comparison(
            results["main_results"],
            metrics,
            title="TreeRAG vs Baselines"
        )
        if path:
            saved_plots["performance_comparison"] = path
        path = plotter.plot_radar_chart(
            results["main_results"],
            metrics
        )
        if path:
            saved_plots["radar_chart"] = path
    
    if "efficiency" in results:
        path = plotter.plot_efficiency_scatter(
            results["efficiency"],
            x_metric="latency_mean_ms",
            y_metric="precision_at_5"
        )
        if path:
            saved_plots["efficiency_scatter"] = path
    
    return saved_plots



# ======================================================================
# PHASE D additions (KCI plan): new CLI entrypoint + helpers.
# Original classes above are retained for backward compatibility.
# ======================================================================
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:  # seaborn optional
    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = _PROJECT_ROOT / "data" / "benchmark_reports"
FIG_DIR = REPORT_DIR / "figures"
LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "flatrag": "FlatRAG",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png / {name}.pdf")


def figure_comparison(report) -> None:
    systems = report["systems"]
    summ = report["summary"]
    labels = [LABELS.get(s, s) for s in systems]
    rouge = [summ[s]["rouge_l"] for s in systems]
    bert = [summ[s]["bertscore"] for s in systems]

    x = range(len(systems))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], rouge, w, label="ROUGE-L", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], bert, w, label="BERTScore", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("System comparison: ROUGE-L vs BERTScore")
    ax.legend()
    _save(fig, "figure_1_comparison")


def figure_ablation(ablation) -> None:
    rows = sorted(ablation["configs"], key=lambda r: r["rouge_l"])
    names = [r["id"] for r in rows]
    scores = [r["rouge_l"] for r in rows]
    deltas = [r["delta_rouge_l_vs_full"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, scores, color="#55A868")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"Δ={d:+.3f}", va="center", fontsize=9)
    ax.set_xlabel("ROUGE-L")
    ax.set_title("Ablation: ROUGE-L by configuration")
    ax.margins(x=0.15)
    _save(fig, "figure_2_ablation")


def figure_efficiency(report) -> None:
    systems = report["systems"]
    summ = report["summary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    pts = []
    for i, s in enumerate(systems):
        a = summ[s]
        ax.scatter(
            a["latency"], a["rouge_l"],
            s=max(60, a["context_tokens"] * 4 + 60),
            color=cmap(i % 10), alpha=0.7, edgecolors="k",
            label=LABELS.get(s, s),
        )
        ax.annotate(LABELS.get(s, s), (a["latency"], a["rouge_l"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
        pts.append((a["latency"], a["rouge_l"], LABELS.get(s, s)))
    frontier, pxs, pys = _pareto_frontier(pts)
    if len(pxs) >= 2:
        ax.plot(pxs, pys, color="crimson", linewidth=1.5, linestyle="--",
                alpha=0.7, label="Pareto frontier")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("Efficiency: Latency vs ROUGE-L — Pareto frontier (upper-left dominates)")
    ax.legend(fontsize=8, loc="best")
    _save(fig, "figure_3_efficiency")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate figures (PHASE D-3)")
    parser.add_argument("--evaluation", default=str(REPORT_DIR / "evaluation_latest.json"))
    parser.add_argument("--ablation", default=str(REPORT_DIR / "ablation_results.json"))
    args = parser.parse_args(argv)

    with open(args.evaluation, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(args.ablation, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    print(f"Generating figures → {FIG_DIR}")
    figure_comparison(report)
    figure_ablation(ablation)
    figure_efficiency(report)
    return 0


# ======================================================================
# PHASE 5 additions (ACM): 4 paper figures, colorblind-safe palette.
# ======================================================================
# Wong (2011) colorblind-safe palette.
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
_ACM_LABELS = {
    "bm25": "BM25", "dense": "Dense", "flatrag": "FlatRAG", "raptor": "RAPTOR",
    "treerag_dfs": "TreeRAG-DFS", "treerag_beam": "TreeRAG-Beam",
}


def _pareto_frontier(points):
    """Return non-dominated points and a staircase path for plotting.

    Assumes lower x (cost) and higher y (quality) is better.
    Returns (frontier_points, xs, ys) where xs/ys are the staircase coordinates.
    """
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_y = float("-inf")
    for x, y, label in sorted_pts:
        if y >= best_y:
            frontier.append((x, y, label))
            best_y = y
    if not frontier:
        return frontier, [], []
    xs, ys = [], []
    for i, (x, y, _) in enumerate(frontier):
        if i == 0:
            xs.append(x); ys.append(y)
        else:
            xs.append(x); ys.append(frontier[i - 1][1])  # horizontal step
            xs.append(x); ys.append(y)                    # vertical step
    return frontier, xs, ys


def figure_architecture(out_dir=None):
    """Figure 1: TreeRAG architecture diagram, rendered to PDF/PNG."""
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    stages = [
        ("PDF\nDocument", 0.06, CB_PALETTE[0]),
        ("LLM Tree\nIndexer", 0.27, CB_PALETTE[1]),
        ("Hierarchical\nPageIndex Tree", 0.50, CB_PALETTE[2]),
        ("Beam/DFS\nTraversal", 0.72, CB_PALETTE[3]),
        ("Grounded Answer\n+ [doc, p.X]", 0.92, CB_PALETTE[4]),
    ]
    for label, x, color in stages:
        ax.add_patch(plt.Rectangle((x - 0.085, 0.4), 0.17, 0.2, facecolor=color,
                                   edgecolor="black", alpha=0.85, transform=ax.transAxes))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", transform=ax.transAxes)
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + 0.085
        x1 = stages[i + 1][1] - 0.085
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5), xycoords=ax.transAxes,
                    textcoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color="black"))
    ax.set_title("TreeRAG: top-down structure-preserving retrieval pipeline")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figure_1_architecture.{0}".format(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figure_1_architecture.pdf / .png")


def figure_main_bars(report, out_dir=None):
    """Figure 2: all systems × {ROUGE-L, BERTScore} grouped bars."""
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    systems = report["systems"]
    labels = [_ACM_LABELS.get(s, s) for s in systems]
    rouge = [report["summary"][s]["rouge_l"] for s in systems]
    bert = [report["summary"][s]["bertscore"] for s in systems]
    x = range(len(systems))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], rouge, w, label="ROUGE-L", color=CB_PALETTE[0])
    ax.bar([i + w / 2 for i in x], bert, w, label="BERTScore", color=CB_PALETTE[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Main results: ROUGE-L and BERTScore by system")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figure_2_main_results.{0}".format(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figure_2_main_results.pdf / .png")


def figure_multihop(full_report, hotpot_report, out_dir=None):
    """Figure 3: Full-benchmark vs HotpotQA ROUGE-L (advantage grows on multi-hop)."""
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    systems = [s for s in full_report["systems"] if s in hotpot_report.get("summary", {})]
    labels = [_ACM_LABELS.get(s, s) for s in systems]
    full = [full_report["summary"][s]["rouge_l"] for s in systems]
    hot = [hotpot_report["summary"][s]["rouge_l"] for s in systems]
    x = range(len(systems))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], full, w, label="Full benchmark", color=CB_PALETTE[2])
    ax.bar([i + w / 2 for i in x], hot, w, label="HotpotQA multi-hop", color=CB_PALETTE[3])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("Multi-hop performance: full benchmark vs HotpotQA subset")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figure_3_multihop.{0}".format(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figure_3_multihop.pdf / .png")


def figure_context_reduction(report, out_dir=None):
    """Figure 4 / Fig 8: accuracy (ROUGE-L) vs context size with Pareto frontier."""
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    systems = report["systems"]
    fig, ax = plt.subplots(figsize=(9, 5))
    pts = []
    for i, s in enumerate(systems):
        a = report["summary"][s]
        marker = "*" if s.startswith("treerag") else "o"
        size = 320 if s.startswith("treerag") else 130
        ax.scatter(a["context_tokens"], a["rouge_l"], s=size, marker=marker,
                   color=CB_PALETTE[i % len(CB_PALETTE)], edgecolors="black",
                   alpha=0.85, label=_ACM_LABELS.get(s, s))
        pts.append((a["context_tokens"], a["rouge_l"], _ACM_LABELS.get(s, s)))
    frontier, pxs, pys = _pareto_frontier(pts)
    if len(pxs) >= 2:
        ax.plot(pxs, pys, color="crimson", linewidth=1.5, linestyle="--",
                alpha=0.7, label="Pareto frontier")
    ax.set_xlabel("Context size (tokens)")
    ax.set_ylabel("ROUGE-L (accuracy)")
    ax.set_title("Accuracy vs context size — Pareto frontier (upper-left dominates)")
    ax.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figure_4_context_reduction.{0}".format(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figure_4_context_reduction.pdf / .png")


if __name__ == "__main__":
    raise SystemExit(main())
