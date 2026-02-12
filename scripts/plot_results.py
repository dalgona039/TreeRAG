
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


if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        sample_results = {
            "Flat RAG": {
                "precision_at_5": 0.65,
                "recall_at_5": 0.72,
                "ndcg_at_5": 0.68,
                "mrr": 0.71,
                "groundedness": 0.82
            },
            "TreeRAG": {
                "precision_at_5": 0.78,
                "recall_at_5": 0.85,
                "ndcg_at_5": 0.81,
                "mrr": 0.84,
                "groundedness": 0.91
            },
            "BM25": {
                "precision_at_5": 0.55,
                "recall_at_5": 0.62,
                "ndcg_at_5": 0.58,
                "mrr": 0.60,
                "groundedness": 0.75
            }
        }
        
        plotter = ResultPlotter("results/figures")
        path = plotter.plot_performance_comparison(
            sample_results,
            ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "groundedness"],
            title="System Comparison"
        )
        print(f"Saved: {path}")
        path = plotter.plot_radar_chart(
            sample_results,
            ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "groundedness"]
        )
        print(f"Saved: {path}")
    else:
        print("matplotlib not available - run 'pip install matplotlib seaborn'")
