
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class TableConfig:
    
    caption: str
    label: str
    column_format: str
    booktabs: bool = True
    centering: bool = True
    float_format: str = ".3f"
    highlight_best: bool = True
    add_notes: bool = True


class LatexTableGenerator:
    
    def __init__(self, output_dir: str = "results/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_main_results_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        config: Optional[TableConfig] = None
    ) -> str:
        if config is None:
            config = TableConfig(
                caption="Main Results Comparison",
                label="tab:main_results",
                column_format="l" + "c" * len(metrics)
            )
        
        best_values = {}
        for metric in metrics:
            values = [results[sys].get(metric, 0) for sys in results]
            best_values[metric] = max(values)
        lines = []
        lines.append("\\begin{table}[htbp]")
        if config.centering:
            lines.append("\\centering")
        lines.append(f"\\caption{{{config.caption}}}")
        lines.append(f"\\label{{{config.label}}}")
        if config.booktabs:
            lines.append(f"\\begin{{tabular}}{{{config.column_format}}}")
            lines.append("\\toprule")
        else:
            lines.append(f"\\begin{{tabular}}{{{config.column_format}}}")
            lines.append("\\hline")
        header = ["System"] + [self._format_metric_name(m) for m in metrics]
        lines.append(" & ".join(header) + " \\\\")
        
        if config.booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")
        for system, system_results in results.items():
            row = [self._escape_latex(system)]
            for metric in metrics:
                value = system_results.get(metric, 0)
                formatted = self._format_value(value, config.float_format)
                if config.highlight_best and abs(value - best_values[metric]) < 1e-6:
                    formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
            
            lines.append(" & ".join(row) + " \\\\")
        if config.booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")
        
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_ablation_table(
        self,
        ablations: Dict[str, Dict[str, float]],
        base_system: str,
        metrics: List[str],
        config: Optional[TableConfig] = None
    ) -> str:
        if config is None:
            config = TableConfig(
                caption="Ablation Study Results",
                label="tab:ablation",
                column_format="l" + "c" * len(metrics) + "c"
            )
        
        base_results = ablations.get(base_system, {})
        
        lines = []
        lines.append("\\begin{table}[htbp]")
        if config.centering:
            lines.append("\\centering")
        lines.append(f"\\caption{{{config.caption}}}")
        lines.append(f"\\label{{{config.label}}}")
        
        col_format = "l" + "c" * len(metrics) + "c"
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")
        header = ["Configuration"] + [self._format_metric_name(m) for m in metrics] + ["$\\Delta$Avg"]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        for name, results in ablations.items():
            row = [self._escape_latex(name)]
            
            deltas = []
            for metric in metrics:
                value = results.get(metric, 0)
                base_value = base_results.get(metric, value)
                
                formatted = self._format_value(value, config.float_format)
                if name != base_system:
                    delta = value - base_value
                    deltas.append(delta)
                    if delta < 0:
                        formatted = f"{formatted} \\textcolor{{red}}{{($\\downarrow$)}}"
                row.append(formatted)
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                delta_str = f"{avg_delta:+.3f}"
                if avg_delta < 0:
                    delta_str = f"\\textcolor{{red}}{{{delta_str}}}"
                else:
                    delta_str = f"\\textcolor{{green}}{{{delta_str}}}"
                row.append(delta_str)
            else:
                row.append("-")
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        
        if config.add_notes:
            lines.append("\\\\[0.5em]")
            lines.append(f"\\footnotesize{{Base system: {base_system}. "
                        "$\\downarrow$ indicates performance drop from removing component.}}")
        
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_significance_table(
        self,
        pairwise_results: Dict[Tuple[str, str], Dict[str, float]],
        systems: List[str],
        metric: str,
        config: Optional[TableConfig] = None
    ) -> str:
        if config is None:
            config = TableConfig(
                caption=f"Statistical Significance Tests ({metric})",
                label=f"tab:significance_{metric}",
                column_format="l" + "c" * len(systems)
            )
        
        lines = []
        lines.append("\\begin{table}[htbp]")
        if config.centering:
            lines.append("\\centering")
        lines.append(f"\\caption{{{config.caption}}}")
        lines.append(f"\\label{{{config.label}}}")
        
        lines.append(f"\\begin{{tabular}}{{{config.column_format}}}")
        lines.append("\\toprule")
        
        header = [""] + [self._escape_latex(s) for s in systems]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        for sys1 in systems:
            row = [self._escape_latex(sys1)]
            for sys2 in systems:
                if sys1 == sys2:
                    row.append("-")
                else:
                    key = (sys1, sys2) if (sys1, sys2) in pairwise_results else (sys2, sys1)
                    if key in pairwise_results:
                        result = pairwise_results[key]
                        p_value = result.get("p_value", 1.0)
                        
                        if p_value < 0.001:
                            cell = "$^{***}$"
                        elif p_value < 0.01:
                            cell = "$^{**}$"
                        elif p_value < 0.05:
                            cell = "$^{*}$"
                        else:
                            cell = f"{p_value:.3f}"
                        
                        row.append(cell)
                    else:
                        row.append("-")
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        
        if config.add_notes:
            lines.append("\\\\[0.5em]")
            lines.append("\\footnotesize{$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$ (paired t-test).}")
        
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_efficiency_table(
        self,
        results: Dict[str, Dict[str, float]],
        config: Optional[TableConfig] = None
    ) -> str:
        if config is None:
            config = TableConfig(
                caption="Efficiency Comparison",
                label="tab:efficiency",
                column_format="lccccc"
            )
        
        metrics = [
            ("latency_mean_ms", "Latency (ms)"),
            ("latency_p95_ms", "P95 Latency"),
            ("tokens_context", "Context Tokens"),
            ("tokens_total", "Total Tokens"),
            ("memory_mb", "Memory (MB)")
        ]
        
        lines = []
        lines.append("\\begin{table}[htbp]")
        if config.centering:
            lines.append("\\centering")
        lines.append(f"\\caption{{{config.caption}}}")
        lines.append(f"\\label{{{config.label}}}")
        
        col_format = "l" + "c" * len(metrics)
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")
        
        header = ["System"] + [name for _, name in metrics]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        best_values = {}
        for key, _ in metrics:
            values = [results[sys].get(key, float('inf')) for sys in results]
            best_values[key] = min(values)
        for system, sys_results in results.items():
            row = [self._escape_latex(system)]
            for key, _ in metrics:
                value = sys_results.get(key, 0)
                if value >= 1000:
                    formatted = f"{value:.0f}"
                elif value >= 100:
                    formatted = f"{value:.1f}"
                else:
                    formatted = f"{value:.2f}"
                if config.highlight_best and abs(value - best_values[key]) < 1e-6:
                    formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def _format_metric_name(self, metric: str) -> str:
        replacements = {
            "precision_at_5": "P@5",
            "recall_at_5": "R@5",
            "f1_at_5": "F1@5",
            "ndcg_at_5": "NDCG@5",
            "mrr": "MRR",
            "map": "MAP",
            "groundedness": "Ground.",
            "hallucination_rate": "Hal.Rate",
            "latency_ms": "Latency",
        }
        return replacements.get(metric, metric.replace("_", " ").title())
    
    def _format_value(self, value: float, fmt: str = ".3f") -> str:
        return f"{value:{fmt}}"
    
    def _escape_latex(self, text: str) -> str:
        replacements = {
            "_": "\\_",
            "&": "\\&",
            "%": "\\%",
            "#": "\\#",
            "$": "\\$",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def save_table(self, table_latex: str, filename: str) -> Path:
        filepath = self.output_dir / filename
        filepath.write_text(table_latex)
        return filepath


def generate_all_tables(results_file: str, output_dir: str = "results/tables") -> Dict[str, Path]:
    with open(results_file) as f:
        results = json.load(f)
    
    generator = LatexTableGenerator(output_dir)
    saved_tables = {}
    
    if "main_results" in results:
        metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "groundedness"]
        table = generator.generate_main_results_table(
            results["main_results"],
            metrics
        )
        path = generator.save_table(table, "main_results.tex")
        saved_tables["main_results"] = path
    
    if "ablation" in results:
        metrics = ["precision_at_5", "ndcg_at_5", "mrr"]
        table = generator.generate_ablation_table(
            results["ablation"],
            "TreeRAG (Full)",
            metrics
        )
        path = generator.save_table(table, "ablation.tex")
        saved_tables["ablation"] = path
    
    if "efficiency" in results:
        table = generator.generate_efficiency_table(results["efficiency"])
        path = generator.save_table(table, "efficiency.tex")
        saved_tables["efficiency"] = path
    
    return saved_tables


if __name__ == "__main__":
    sample_results = {
        "main_results": {
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
            }
        },
        "ablation": {
            "TreeRAG (Full)": {
                "precision_at_5": 0.78,
                "ndcg_at_5": 0.81,
                "mrr": 0.84
            },
            "- Hierarchical Index": {
                "precision_at_5": 0.68,
                "ndcg_at_5": 0.72,
                "mrr": 0.75
            },
            "- Beam Search": {
                "precision_at_5": 0.73,
                "ndcg_at_5": 0.77,
                "mrr": 0.80
            }
        }
    }
    
    generator = LatexTableGenerator("results/tables")
    table = generator.generate_main_results_table(
        sample_results["main_results"],
        ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "groundedness"]
    )
    print("Main Results Table:")
    print(table)
    print()
    ablation_table = generator.generate_ablation_table(
        sample_results["ablation"],
        "TreeRAG (Full)",
        ["precision_at_5", "ndcg_at_5", "mrr"]
    )
    print("Ablation Table:")
    print(ablation_table)
