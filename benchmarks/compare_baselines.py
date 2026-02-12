"""
Compare Baselines: FlatRAG vs TreeRAG

This module runs comprehensive comparisons between TreeRAG
and baseline retrieval systems with statistical testing.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from enum import Enum

from benchmarks.metrics.retrieval_metrics import (
    RetrievalMetrics, QueryResult, RetrievalResult, 
    BenchmarkMetrics, create_query_result
)
from benchmarks.metrics.efficiency_metrics import (
    EfficiencyMetrics, LatencyMeasurement, TokenUsage, 
    TraversalStats, EfficiencyResult
)
from benchmarks.metrics.fidelity_metrics import (
    FidelityMetrics, FidelityAnalysis, FidelityResult
)
from benchmarks.metrics.statistical_tests import (
    StatisticalTests, ComparisonSummary, generate_latex_table
)


class BaselineType(str, Enum):
    """Types of baseline systems."""
    FLAT_RAG = "flat_rag"
    TREE_RAG = "tree_rag"
    NAIVE_CHUNKING = "naive_chunking"
    BM25 = "bm25"
    DENSE_RETRIEVAL = "dense_retrieval"
    HYBRID = "hybrid"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Metrics to compute
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    compute_retrieval: bool = True
    compute_efficiency: bool = True
    compute_fidelity: bool = True
    
    # Statistical testing
    alpha: float = 0.05
    n_bootstrap: int = 10000
    
    # Output format
    output_latex: bool = True
    output_json: bool = True
    output_dir: str = "benchmarks/results"


@dataclass
class SystemResult:
    """Results for a single system."""
    system_name: str
    system_type: BaselineType
    retrieval_metrics: Optional[BenchmarkMetrics] = None
    efficiency_metrics: Optional[EfficiencyResult] = None
    fidelity_metrics: Optional[FidelityResult] = None
    
    # Raw per-query data for statistical testing
    per_query_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_name": self.system_name,
            "system_type": self.system_type.value,
            "retrieval": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "efficiency": self.efficiency_metrics.to_dict() if self.efficiency_metrics else None,
            "fidelity": self.fidelity_metrics.to_dict() if self.fidelity_metrics else None,
            "per_query": self.per_query_scores
        }


@dataclass
class ComparisonResult:
    """Result of comparing two systems."""
    primary: SystemResult
    baseline: SystemResult
    statistical_comparisons: Dict[str, ComparisonSummary] = field(default_factory=dict)
    
    # Summary
    primary_wins: int = 0
    baseline_wins: int = 0
    ties: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary": self.primary.to_dict(),
            "baseline": self.baseline.to_dict(),
            "statistical_comparisons": {
                name: comp.to_dict() 
                for name, comp in self.statistical_comparisons.items()
            },
            "summary": {
                "primary_wins": self.primary_wins,
                "baseline_wins": self.baseline_wins,
                "ties": self.ties
            }
        }


class BaselineRunner:
    """
    Runs baseline retrieval systems for comparison.
    
    Implements multiple retrieval strategies:
    - FlatRAG: Traditional chunk-based retrieval
    - TreeRAG: Hierarchical tree-based retrieval
    - Naive chunking: Fixed-size chunks
    - BM25: Term-based retrieval
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize baseline runner."""
        self.config = config or BenchmarkConfig()
        
        # Metrics calculators
        self.retrieval_metrics = RetrievalMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.fidelity_metrics = FidelityMetrics()
        self.stats = StatisticalTests(alpha=self.config.alpha)
    
    def run_flat_rag(
        self,
        queries: List[str],
        documents: List[str],
        chunk_size: int = 512,
        top_k: int = 5
    ) -> SystemResult:
        """
        Run FlatRAG baseline.
        
        Chunks documents into fixed-size pieces and retrieves
        using dense embedding similarity.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            chunk_size: Size of chunks in characters
            top_k: Number of results to retrieve
            
        Returns:
            System results
        """
        result = SystemResult(
            system_name="FlatRAG",
            system_type=BaselineType.FLAT_RAG
        )
        
        # Chunk documents
        chunks = self._chunk_documents(documents, chunk_size)
        
        query_results = []
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            
            # Simple similarity-based retrieval (placeholder)
            # In real implementation, use embedding similarity
            retrieved = self._retrieve_chunks(query, chunks, top_k)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Create query result (ground truth would come from labeled data)
            qr = QueryResult(
                query_id=f"q_{i}",
                query_text=query,
                retrieved=retrieved,
                relevant_doc_ids=set(),  # Would be from labeled data
                latency_ms=latency_ms
            )
            query_results.append(qr)
            
            # Store per-query latency
            result.per_query_scores[f"q_{i}"] = {"latency": latency_ms}
        
        return result
    
    def _chunk_documents(
        self, 
        documents: List[str], 
        chunk_size: int
    ) -> List[tuple]:
        """Chunk documents into fixed-size pieces."""
        chunks = []
        for doc_idx, doc in enumerate(documents):
            for i in range(0, len(doc), chunk_size):
                chunk_text = doc[i:i + chunk_size]
                chunk_id = f"doc_{doc_idx}_chunk_{i // chunk_size}"
                chunks.append((chunk_id, chunk_text))
        return chunks
    
    def _retrieve_chunks(
        self,
        query: str,
        chunks: List[tuple],
        top_k: int
    ) -> List[RetrievalResult]:
        """Retrieve top-k chunks (placeholder using simple matching)."""
        # Simple word overlap scoring (placeholder for embedding similarity)
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk_id, chunk_text in chunks:
            chunk_words = set(chunk_text.lower().split())
            overlap = len(query_words & chunk_words)
            score = overlap / len(query_words) if query_words else 0
            scored_chunks.append((chunk_id, score, chunk_text))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk_id, score, _) in enumerate(scored_chunks[:top_k], 1):
            results.append(RetrievalResult(
                doc_id=chunk_id,
                rank=rank,
                score=score,
                relevance=0.0  # Would be from labeled data
            ))
        
        return results


class BaselineComparison:
    """
    Compare TreeRAG against baseline systems.
    
    Provides comprehensive comparison with:
    - Retrieval quality metrics
    - Efficiency metrics
    - Fidelity metrics
    - Statistical significance testing
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize comparison."""
        self.config = config or BenchmarkConfig()
        self.stats = StatisticalTests(alpha=self.config.alpha)
    
    def compare(
        self,
        treerag_result: SystemResult,
        baseline_result: SystemResult
    ) -> ComparisonResult:
        """
        Compare TreeRAG against a baseline.
        
        Args:
            treerag_result: Results from TreeRAG
            baseline_result: Results from baseline system
            
        Returns:
            Comparison result with statistical tests
        """
        comparison = ComparisonResult(
            primary=treerag_result,
            baseline=baseline_result
        )
        
        # Compare retrieval metrics
        if treerag_result.retrieval_metrics and baseline_result.retrieval_metrics:
            self._compare_retrieval(comparison)
        
        # Compare efficiency metrics
        if treerag_result.efficiency_metrics and baseline_result.efficiency_metrics:
            self._compare_efficiency(comparison)
        
        # Compare fidelity metrics
        if treerag_result.fidelity_metrics and baseline_result.fidelity_metrics:
            self._compare_fidelity(comparison)
        
        # Count wins
        for name, comp in comparison.statistical_comparisons.items():
            if comp.winner == treerag_result.system_name:
                comparison.primary_wins += 1
            elif comp.winner == baseline_result.system_name:
                comparison.baseline_wins += 1
            else:
                comparison.ties += 1
        
        return comparison
    
    def _compare_retrieval(self, comparison: ComparisonResult) -> None:
        """Compare retrieval metrics with statistical tests."""
        treerag = comparison.primary.retrieval_metrics
        baseline = comparison.baseline.retrieval_metrics
        
        # Compare at each K value
        for k in self.config.k_values:
            # P@K
            if k in treerag.precision_at_k and k in baseline.precision_at_k:
                treerag_scores = list(
                    m.get(f"P@{k}", 0.0) 
                    for m in treerag.per_query_metrics.values()
                )
                baseline_scores = list(
                    m.get(f"P@{k}", 0.0) 
                    for m in baseline.per_query_metrics.values()
                )
                
                if treerag_scores and baseline_scores:
                    comp = self.stats.compare_methods(
                        comparison.primary.system_name,
                        comparison.baseline.system_name,
                        treerag_scores,
                        baseline_scores,
                        f"P@{k}"
                    )
                    comparison.statistical_comparisons[f"P@{k}"] = comp
            
            # NDCG@K
            if k in treerag.ndcg_at_k and k in baseline.ndcg_at_k:
                treerag_scores = list(
                    m.get(f"NDCG@{k}", 0.0) 
                    for m in treerag.per_query_metrics.values()
                )
                baseline_scores = list(
                    m.get(f"NDCG@{k}", 0.0) 
                    for m in baseline.per_query_metrics.values()
                )
                
                if treerag_scores and baseline_scores:
                    comp = self.stats.compare_methods(
                        comparison.primary.system_name,
                        comparison.baseline.system_name,
                        treerag_scores,
                        baseline_scores,
                        f"NDCG@{k}"
                    )
                    comparison.statistical_comparisons[f"NDCG@{k}"] = comp
        
        # MRR
        treerag_mrr = list(
            m.get("MRR", 0.0) for m in treerag.per_query_metrics.values()
        )
        baseline_mrr = list(
            m.get("MRR", 0.0) for m in baseline.per_query_metrics.values()
        )
        
        if treerag_mrr and baseline_mrr:
            comp = self.stats.compare_methods(
                comparison.primary.system_name,
                comparison.baseline.system_name,
                treerag_mrr,
                baseline_mrr,
                "MRR"
            )
            comparison.statistical_comparisons["MRR"] = comp
    
    def _compare_efficiency(self, comparison: ComparisonResult) -> None:
        """Compare efficiency metrics."""
        treerag = comparison.primary.efficiency_metrics
        baseline = comparison.baseline.efficiency_metrics
        
        # Latency comparison
        treerag_latencies = list(treerag.per_query_latencies.values())
        baseline_latencies = list(baseline.per_query_latencies.values())
        
        if treerag_latencies and baseline_latencies:
            comp = self.stats.compare_methods(
                comparison.baseline.system_name,  # Lower is better, so swap
                comparison.primary.system_name,
                baseline_latencies,
                treerag_latencies,
                "Latency (ms)"
            )
            comparison.statistical_comparisons["Latency"] = comp
        
        # Token usage comparison
        treerag_tokens = list(treerag.per_query_tokens.values())
        baseline_tokens = list(baseline.per_query_tokens.values())
        
        if treerag_tokens and baseline_tokens:
            comp = self.stats.compare_methods(
                comparison.baseline.system_name,  # Lower is better
                comparison.primary.system_name,
                baseline_tokens,
                treerag_tokens,
                "Tokens"
            )
            comparison.statistical_comparisons["Tokens"] = comp
    
    def _compare_fidelity(self, comparison: ComparisonResult) -> None:
        """Compare fidelity metrics."""
        treerag = comparison.primary.fidelity_metrics
        baseline = comparison.baseline.fidelity_metrics
        
        # Groundedness comparison
        treerag_groundedness = list(treerag.per_query_groundedness.values())
        baseline_groundedness = list(baseline.per_query_groundedness.values())
        
        if treerag_groundedness and baseline_groundedness:
            comp = self.stats.compare_methods(
                comparison.primary.system_name,
                comparison.baseline.system_name,
                treerag_groundedness,
                baseline_groundedness,
                "Groundedness"
            )
            comparison.statistical_comparisons["Groundedness"] = comp
    
    def generate_report(
        self,
        comparison: ComparisonResult
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            comparison: Comparison result
            
        Returns:
            Report dictionary
        """
        report = {
            "systems": {
                "primary": comparison.primary.system_name,
                "baseline": comparison.baseline.system_name
            },
            "summary": {
                f"{comparison.primary.system_name}_wins": comparison.primary_wins,
                f"{comparison.baseline.system_name}_wins": comparison.baseline_wins,
                "ties": comparison.ties,
                "total_comparisons": len(comparison.statistical_comparisons)
            },
            "comparisons": {}
        }
        
        for name, comp in comparison.statistical_comparisons.items():
            ttest = comp.tests.get("paired_ttest")
            bootstrap = comp.tests.get("bootstrap")
            
            report["comparisons"][name] = {
                "winner": comp.winner,
                "confidence": comp.confidence,
                f"{comp.method_a}_mean": comp.method_a_mean,
                f"{comp.method_b}_mean": comp.method_b_mean,
                "difference": comp.method_a_mean - comp.method_b_mean,
                "p_value": ttest.p_value if ttest else None,
                "effect_size": ttest.effect_size if ttest else None,
                "effect_interpretation": ttest.effect_size_interpretation if ttest else None,
                "ci_95": [bootstrap.ci_lower, bootstrap.ci_upper] if bootstrap else None
            }
        
        return report
    
    def generate_latex(self, comparison: ComparisonResult) -> str:
        """Generate LaTeX table for paper."""
        return generate_latex_table(
            list(comparison.statistical_comparisons.values())
        )


def run_full_comparison(
    treerag_query_results: List[QueryResult],
    flatrag_query_results: List[QueryResult],
    config: Optional[BenchmarkConfig] = None
) -> ComparisonResult:
    """
    Run full comparison between TreeRAG and FlatRAG.
    
    This is the main entry point for benchmarking.
    
    Args:
        treerag_query_results: Query results from TreeRAG
        flatrag_query_results: Query results from FlatRAG
        config: Benchmark configuration
        
    Returns:
        Comprehensive comparison result
    """
    config = config or BenchmarkConfig()
    
    # Compute metrics
    retrieval_calc = RetrievalMetrics()
    
    treerag_metrics = retrieval_calc.compute_all_metrics(
        treerag_query_results, config.k_values
    )
    flatrag_metrics = retrieval_calc.compute_all_metrics(
        flatrag_query_results, config.k_values
    )
    
    # Create system results
    treerag_result = SystemResult(
        system_name="TreeRAG",
        system_type=BaselineType.TREE_RAG,
        retrieval_metrics=treerag_metrics
    )
    
    flatrag_result = SystemResult(
        system_name="FlatRAG",
        system_type=BaselineType.FLAT_RAG,
        retrieval_metrics=flatrag_metrics
    )
    
    # Compare
    comparator = BaselineComparison(config)
    return comparator.compare(treerag_result, flatrag_result)


def save_results(
    comparison: ComparisonResult,
    output_dir: str = "benchmarks/results"
) -> None:
    """
    Save comparison results to files.
    
    Args:
        comparison: Comparison result
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_path / "comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Save LaTeX
    comparator = BaselineComparison()
    latex = comparator.generate_latex(comparison)
    
    latex_path = output_path / "comparison_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    
    # Save report
    report = comparator.generate_report(comparison)
    
    report_path = output_path / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
