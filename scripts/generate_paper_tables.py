
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



# ======================================================================
# PHASE D additions (KCI plan): new CLI entrypoint + helpers.
# Original classes above are retained for backward compatibility.
# ======================================================================
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REPORT_DIR = _PROJECT_ROOT / "data" / "benchmark_reports"
INDEX_DIR = _PROJECT_ROOT / "data" / "indices"
LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "flatrag": "FlatRAG",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}


def _esc(text: str) -> str:
    return text.replace("_", r"\_").replace("#", r"\#").replace("&", r"\&")


def _best_index(values: List[Optional[float]], maximize: bool = True) -> Optional[int]:
    vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not vals:
        return None
    return (max if maximize else min)(vals, key=lambda x: x[1])[0]


def _fmt(value: Optional[float], best: bool, prec: int = 3, suffix: str = "") -> str:
    if value is None:
        return "--"
    s = f"{value:.{prec}f}{suffix}"
    return rf"\textbf{{{s}}}" if best else s


# --------------------------------------------------------------------------- #
def table_main(report: Dict[str, Any]) -> str:
    systems = report["systems"]
    summ = report["summary"]
    sig = report.get("significance", {})

    cols = {
        "rouge_l": ("ROUGE-L", True, 3, ""),
        "bertscore": ("BERTScore", True, 3, ""),
        "llm_judge": ("LLM-Judge", True, 2, ""),
        "latency": ("Latency (s)", False, 3, ""),
        "context_tokens": ("CTX (K)", False, 1, ""),
    }
    best = {}
    for key, (_, maximize, _, _) in cols.items():
        vals = [summ[s].get(key) for s in systems]
        if key == "context_tokens":
            vals = [v / 1000 if v is not None else None for v in vals]
        best[key] = _best_index(vals, maximize)

    lines = [
        r"% Table 1: Main system comparison",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main comparison across retrieval systems. Best per column in \textbf{bold}.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & ROUGE-L & BERTScore & LLM-Judge & Latency (s) & CTX (K) \\",
        r"\midrule",
    ]
    for i, s in enumerate(systems):
        ctx = summ[s]["context_tokens"] / 1000
        row = " & ".join(
            [
                _esc(LABELS.get(s, s)),
                _fmt(summ[s]["rouge_l"], best["rouge_l"] == i, 3),
                _fmt(summ[s]["bertscore"], best["bertscore"] == i, 3),
                _fmt(summ[s].get("llm_judge"), best["llm_judge"] == i, 2),
                _fmt(summ[s]["latency"], best["latency"] == i, 3),
                _fmt(ctx, best["context_tokens"] == i, 1),
            ]
        )
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    if sig:
        notes = []
        for s, r in sig.items():
            mark = "$^{*}$" if r.get("significant") else ""
            notes.append(f"{_esc(LABELS.get(s, s))}: $p={r['p_value']:.4f}${mark}")
        lines.append(
            r"\\[2pt]\footnotesize{Paired $t$-test vs TreeRAG-Beam (ROUGE-L). "
            + "; ".join(notes)
            + r". $^{*}p<0.05$.}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_ablation(ablation: Dict[str, Any]) -> str:
    rows = ablation["configs"]
    best_i = _best_index([r["rouge_l"] for r in rows], True)
    lines = [
        r"% Table 2: Ablation study",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study. $\Delta$ is ROUGE-L relative to the full system.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & ROUGE-L & $\Delta$ vs Full & CTX Reduction \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(
            " & ".join(
                [
                    _esc(r["id"]),
                    _fmt(r["rouge_l"], best_i == i, 3),
                    f"{r['delta_rouge_l_vs_full']:+.3f}",
                    f"{r['ctx_reduction_vs_full'] * 100:+.1f}\\%",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _doc_pages(doc_id: str) -> int:
    path = INDEX_DIR / doc_id
    if not path.exists():
        path = INDEX_DIR / f"{doc_id}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
        ref = str(root.get("page_ref", "")).replace("p.", "")
        nums = [int(x) for x in ref.replace("-", " ").split() if x.isdigit()]
        return max(nums) if nums else 0
    except Exception:
        return 0


def table_efficiency(report: Dict[str, Any]) -> str:
    pq = report.get("per_question", {})
    if "treerag_beam" not in pq or "flatrag" not in pq:
        return "% Table 3 skipped: requires treerag_beam and flatrag systems.\n"

    beam = {r["question_id"]: r for r in pq["treerag_beam"]}
    flat = {r["question_id"]: r for r in pq["flatrag"]}
    buckets = {"<50p": [], "50-100p": [], ">100p": []}
    for qid, b in beam.items():
        if qid not in flat:
            continue
        pages = _doc_pages(b["document_id"])
        key = "<50p" if pages < 50 else ("50-100p" if pages <= 100 else ">100p")
        f_ctx = flat[qid]["context_tokens"]
        red = (f_ctx - b["context_tokens"]) / f_ctx if f_ctx else 0.0
        buckets[key].append(red)

    lines = [
        r"% Table 3: Efficiency by document size",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Context reduction of TreeRAG-Beam vs FlatRAG by document size.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Document size & \# Questions & CTX Reduction \\",
        r"\midrule",
    ]
    for key, vals in buckets.items():
        if not vals:
            lines.append(f"{key} & 0 & -- \\\\")
        else:
            avg = sum(vals) / len(vals)
            lines.append(f"{key} & {len(vals)} & {avg * 100:+.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables (PHASE D-2)")
    parser.add_argument("--evaluation", default=str(REPORT_DIR / "evaluation_latest.json"))
    parser.add_argument("--ablation", default=str(REPORT_DIR / "ablation_results.json"))
    parser.add_argument("--output", default=str(REPORT_DIR / "paper_tables.tex"))
    args = parser.parse_args(argv)

    with open(args.evaluation, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(args.ablation, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    blocks = [
        f"% Auto-generated from {os.path.basename(args.evaluation)} "
        f"(mode={report.get('mode')}) and {os.path.basename(args.ablation)}.",
        f"% NOTE: regenerate after an ONLINE run for publication numbers.",
        table_main(report),
        "",
        table_ablation(ablation),
        "",
        table_efficiency(report),
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"💾 LaTeX tables → {out}")
    print("\n" + "\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
