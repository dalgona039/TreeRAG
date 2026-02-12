"""
Efficiency Metrics for TreeRAG Benchmarking.

Measures computational efficiency:
- Token reduction rate (context compression)
- Latency (query processing time)
- Throughput (queries per second)
- Memory footprint
- Tree traversal statistics
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum


class EfficiencyMetricType(str, Enum):
    """Types of efficiency metrics."""
    TOKEN_REDUCTION = "token_reduction"
    LATENCY_MS = "latency_ms"
    THROUGHPUT_QPS = "throughput_qps"
    NODES_VISITED = "nodes_visited"
    TREE_DEPTH_USED = "tree_depth_used"
    CONTEXT_LENGTH = "context_length"
    API_CALLS = "api_calls"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    query_id: str
    total_ms: float
    indexing_ms: float = 0.0
    traversal_ms: float = 0.0
    llm_ms: float = 0.0
    embedding_ms: float = 0.0
    
    @property
    def overhead_ms(self) -> float:
        """Non-LLM overhead time."""
        return self.total_ms - self.llm_ms


@dataclass
class TokenUsage:
    """Token usage statistics."""
    query_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_tokens: int  # Tokens in retrieved context
    original_document_tokens: int  # Full document tokens
    
    @property
    def reduction_rate(self) -> float:
        """Token reduction rate (1 - compressed/original)."""
        if self.original_document_tokens == 0:
            return 0.0
        return 1.0 - (self.context_tokens / self.original_document_tokens)
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (original/compressed)."""
        if self.context_tokens == 0:
            return float('inf')
        return self.original_document_tokens / self.context_tokens


@dataclass
class TraversalStats:
    """Tree traversal statistics."""
    query_id: str
    nodes_visited: int
    nodes_pruned: int
    max_depth_reached: int
    tree_total_nodes: int
    tree_max_depth: int
    beam_width_used: int = 1
    
    @property
    def visit_rate(self) -> float:
        """Fraction of tree visited."""
        if self.tree_total_nodes == 0:
            return 0.0
        return self.nodes_visited / self.tree_total_nodes
    
    @property
    def pruning_rate(self) -> float:
        """Fraction of nodes pruned."""
        total_considered = self.nodes_visited + self.nodes_pruned
        if total_considered == 0:
            return 0.0
        return self.nodes_pruned / total_considered


@dataclass
class EfficiencyResult:
    """Aggregated efficiency metrics."""
    # Latency statistics
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0
    
    # LLM-specific latency
    llm_latency_mean_ms: float = 0.0
    overhead_mean_ms: float = 0.0
    
    # Throughput
    throughput_qps: float = 0.0
    
    # Token efficiency
    token_reduction_mean: float = 0.0
    token_reduction_std: float = 0.0
    compression_ratio_mean: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Traversal efficiency
    nodes_visited_mean: float = 0.0
    visit_rate_mean: float = 0.0
    pruning_rate_mean: float = 0.0
    max_depth_mean: float = 0.0
    
    # Per-query details
    per_query_latencies: Dict[str, float] = field(default_factory=dict)
    per_query_tokens: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "median_ms": self.latency_median_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
                "std_ms": self.latency_std_ms,
                "llm_mean_ms": self.llm_latency_mean_ms,
                "overhead_mean_ms": self.overhead_mean_ms
            },
            "throughput_qps": self.throughput_qps,
            "tokens": {
                "reduction_mean": self.token_reduction_mean,
                "reduction_std": self.token_reduction_std,
                "compression_ratio_mean": self.compression_ratio_mean,
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens
            },
            "traversal": {
                "nodes_visited_mean": self.nodes_visited_mean,
                "visit_rate_mean": self.visit_rate_mean,
                "pruning_rate_mean": self.pruning_rate_mean,
                "max_depth_mean": self.max_depth_mean
            }
        }


class EfficiencyMetrics:
    """
    Calculator for efficiency metrics.
    
    Measures token reduction, latency, and traversal efficiency
    for TreeRAG systems compared to baselines.
    """
    
    def __init__(self):
        """Initialize efficiency metrics calculator."""
        self._latency_measurements: List[LatencyMeasurement] = []
        self._token_usage: List[TokenUsage] = []
        self._traversal_stats: List[TraversalStats] = []
    
    def record_latency(self, measurement: LatencyMeasurement) -> None:
        """Record a latency measurement."""
        self._latency_measurements.append(measurement)
    
    def record_tokens(self, usage: TokenUsage) -> None:
        """Record token usage."""
        self._token_usage.append(usage)
    
    def record_traversal(self, stats: TraversalStats) -> None:
        """Record traversal statistics."""
        self._traversal_stats.append(stats)
    
    def clear(self) -> None:
        """Clear all recorded measurements."""
        self._latency_measurements.clear()
        self._token_usage.clear()
        self._traversal_stats.clear()
    
    def percentile(self, values: List[float], p: float) -> float:
        """Compute percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        
        if f == c:
            return sorted_values[int(k)]
        
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
    
    def compute_latency_stats(self) -> Dict[str, float]:
        """Compute latency statistics."""
        if not self._latency_measurements:
            return {}
        
        total_times = [m.total_ms for m in self._latency_measurements]
        llm_times = [m.llm_ms for m in self._latency_measurements]
        overhead_times = [m.overhead_ms for m in self._latency_measurements]
        
        return {
            "mean_ms": statistics.mean(total_times),
            "median_ms": statistics.median(total_times),
            "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            "p95_ms": self.percentile(total_times, 95),
            "p99_ms": self.percentile(total_times, 99),
            "llm_mean_ms": statistics.mean(llm_times),
            "overhead_mean_ms": statistics.mean(overhead_times)
        }
    
    def compute_token_stats(self) -> Dict[str, float]:
        """Compute token usage statistics."""
        if not self._token_usage:
            return {}
        
        reductions = [u.reduction_rate for u in self._token_usage]
        compressions = [u.compression_ratio for u in self._token_usage if u.compression_ratio != float('inf')]
        
        return {
            "reduction_mean": statistics.mean(reductions),
            "reduction_std": statistics.stdev(reductions) if len(reductions) > 1 else 0.0,
            "compression_ratio_mean": statistics.mean(compressions) if compressions else 0.0,
            "total_input": sum(u.input_tokens for u in self._token_usage),
            "total_output": sum(u.output_tokens for u in self._token_usage),
            "total_context": sum(u.context_tokens for u in self._token_usage)
        }
    
    def compute_traversal_stats(self) -> Dict[str, float]:
        """Compute traversal statistics."""
        if not self._traversal_stats:
            return {}
        
        return {
            "nodes_visited_mean": statistics.mean(s.nodes_visited for s in self._traversal_stats),
            "visit_rate_mean": statistics.mean(s.visit_rate for s in self._traversal_stats),
            "pruning_rate_mean": statistics.mean(s.pruning_rate for s in self._traversal_stats),
            "max_depth_mean": statistics.mean(s.max_depth_reached for s in self._traversal_stats)
        }
    
    def compute_all(self) -> EfficiencyResult:
        """Compute all efficiency metrics."""
        result = EfficiencyResult()
        
        # Latency stats
        latency_stats = self.compute_latency_stats()
        if latency_stats:
            result.latency_mean_ms = latency_stats["mean_ms"]
            result.latency_median_ms = latency_stats["median_ms"]
            result.latency_std_ms = latency_stats["std_ms"]
            result.latency_p95_ms = latency_stats["p95_ms"]
            result.latency_p99_ms = latency_stats["p99_ms"]
            result.llm_latency_mean_ms = latency_stats["llm_mean_ms"]
            result.overhead_mean_ms = latency_stats["overhead_mean_ms"]
            
            # Compute throughput
            total_time_s = sum(m.total_ms for m in self._latency_measurements) / 1000
            if total_time_s > 0:
                result.throughput_qps = len(self._latency_measurements) / total_time_s
            
            # Per-query latencies
            for m in self._latency_measurements:
                result.per_query_latencies[m.query_id] = m.total_ms
        
        # Token stats
        token_stats = self.compute_token_stats()
        if token_stats:
            result.token_reduction_mean = token_stats["reduction_mean"]
            result.token_reduction_std = token_stats["reduction_std"]
            result.compression_ratio_mean = token_stats["compression_ratio_mean"]
            result.total_input_tokens = token_stats["total_input"]
            result.total_output_tokens = token_stats["total_output"]
            
            # Per-query tokens
            for u in self._token_usage:
                result.per_query_tokens[u.query_id] = u.total_tokens
        
        # Traversal stats
        traversal_stats = self.compute_traversal_stats()
        if traversal_stats:
            result.nodes_visited_mean = traversal_stats["nodes_visited_mean"]
            result.visit_rate_mean = traversal_stats["visit_rate_mean"]
            result.pruning_rate_mean = traversal_stats["pruning_rate_mean"]
            result.max_depth_mean = traversal_stats["max_depth_mean"]
        
        return result
    
    @staticmethod
    def measure_latency(func: Callable, *args, **kwargs) -> tuple:
        """
        Measure latency of a function call.
        
        Args:
            func: Function to measure
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass
            
        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        return result, latency_ms
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count for text.
        
        Uses tiktoken if available, otherwise estimates.
        
        Args:
            text: Text to count tokens for
            model: Model name for tokenizer
            
        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except ImportError:
            # Rough estimate: ~4 characters per token
            return len(text) // 4


class LatencyTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latency_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        return False


def compare_token_efficiency(
    treerag_tokens: List[int],
    baseline_tokens: List[int]
) -> Dict[str, float]:
    """
    Compare token efficiency between TreeRAG and baseline.
    
    Args:
        treerag_tokens: Token counts for TreeRAG queries
        baseline_tokens: Token counts for baseline queries (same order)
        
    Returns:
        Comparison statistics
    """
    if not treerag_tokens or not baseline_tokens:
        return {}
    
    if len(treerag_tokens) != len(baseline_tokens):
        raise ValueError("Token lists must have same length")
    
    # Per-query reduction
    reductions = [
        1.0 - (tr / bl) if bl > 0 else 0.0
        for tr, bl in zip(treerag_tokens, baseline_tokens)
    ]
    
    return {
        "treerag_total": sum(treerag_tokens),
        "baseline_total": sum(baseline_tokens),
        "total_reduction": 1.0 - (sum(treerag_tokens) / sum(baseline_tokens)),
        "mean_reduction": statistics.mean(reductions),
        "median_reduction": statistics.median(reductions),
        "min_reduction": min(reductions),
        "max_reduction": max(reductions)
    }


def compare_latency(
    treerag_latencies: List[float],
    baseline_latencies: List[float]
) -> Dict[str, float]:
    """
    Compare latency between TreeRAG and baseline.
    
    Args:
        treerag_latencies: Latencies for TreeRAG queries (ms)
        baseline_latencies: Latencies for baseline queries (ms)
        
    Returns:
        Comparison statistics
    """
    if not treerag_latencies or not baseline_latencies:
        return {}
    
    return {
        "treerag_mean_ms": statistics.mean(treerag_latencies),
        "baseline_mean_ms": statistics.mean(baseline_latencies),
        "treerag_p95_ms": sorted(treerag_latencies)[int(len(treerag_latencies) * 0.95)],
        "baseline_p95_ms": sorted(baseline_latencies)[int(len(baseline_latencies) * 0.95)],
        "speedup": statistics.mean(baseline_latencies) / statistics.mean(treerag_latencies),
        "latency_difference_ms": statistics.mean(treerag_latencies) - statistics.mean(baseline_latencies)
    }
