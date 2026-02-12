"""
Retrieval Metrics for TreeRAG Benchmarking.

Implements standard Information Retrieval metrics:
- Precision@K (P@K)
- Recall@K (R@K) 
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR)
- Hit Rate@K (HR@K)
- Mean Average Precision (MAP)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class MetricType(str, Enum):
    """Types of retrieval metrics."""
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    NDCG_AT_K = "ndcg_at_k"
    MRR = "mrr"
    HIT_RATE_AT_K = "hit_rate_at_k"
    MAP = "map"
    F1_AT_K = "f1_at_k"


@dataclass
class RetrievalResult:
    """Single retrieval result with relevance information."""
    doc_id: str
    rank: int
    score: float
    relevance: float  # 0.0 to 1.0 (binary or graded)
    
    def is_relevant(self, threshold: float = 0.5) -> bool:
        """Check if result is relevant above threshold."""
        return self.relevance >= threshold


@dataclass
class QueryResult:
    """Results for a single query."""
    query_id: str
    query_text: str
    retrieved: List[RetrievalResult]
    relevant_doc_ids: Set[str]  # Ground truth
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass 
class MetricResult:
    """Result of computing a metric."""
    metric_type: MetricType
    value: float
    k: Optional[int] = None
    query_id: Optional[str] = None
    
    def __str__(self) -> str:
        k_str = f"@{self.k}" if self.k else ""
        return f"{self.metric_type.value}{k_str}: {self.value:.4f}"


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics across all queries."""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Per-query details for statistical tests
    per_query_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision@k": self.precision_at_k,
            "recall@k": self.recall_at_k,
            "ndcg@k": self.ndcg_at_k,
            "mrr": self.mrr,
            "map": self.map_score,
            "hit_rate@k": self.hit_rate_at_k,
            "f1@k": self.f1_at_k,
            "per_query": self.per_query_metrics
        }


class RetrievalMetrics:
    """
    Calculator for standard retrieval metrics.
    
    Supports both binary and graded relevance judgments.
    All metrics follow standard IR definitions from:
    - Manning et al., "Introduction to Information Retrieval"
    - Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation"
    """
    
    def __init__(self, relevance_threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            relevance_threshold: Threshold for binary relevance (default: 0.5)
        """
        self.relevance_threshold = relevance_threshold
    
    def precision_at_k(
        self, 
        retrieved: List[RetrievalResult], 
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        P@K = |{relevant docs in top-K}| / K
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Cutoff rank
            
        Returns:
            Precision at rank K
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(
            1 for r in top_k if r.doc_id in relevant_ids
        )
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        retrieved: List[RetrievalResult],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        R@K = |{relevant docs in top-K}| / |{all relevant docs}|
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Cutoff rank
            
        Returns:
            Recall at rank K
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(
            1 for r in top_k if r.doc_id in relevant_ids
        )
        
        return relevant_in_top_k / len(relevant_ids)
    
    def f1_at_k(
        self,
        retrieved: List[RetrievalResult],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Compute F1@K (harmonic mean of P@K and R@K).
        
        F1@K = 2 * P@K * R@K / (P@K + R@K)
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs
            k: Cutoff rank
            
        Returns:
            F1 score at rank K
        """
        p = self.precision_at_k(retrieved, relevant_ids, k)
        r = self.recall_at_k(retrieved, relevant_ids, k)
        
        if p + r == 0:
            return 0.0
        
        return 2 * p * r / (p + r)
    
    def dcg_at_k(
        self,
        retrieved: List[RetrievalResult],
        k: int,
        use_graded: bool = True
    ) -> float:
        """
        Compute Discounted Cumulative Gain at K.
        
        DCG@K = Σ(i=1 to K) rel_i / log2(i+1)
        
        Alternative formula (used here):
        DCG@K = Σ(i=1 to K) (2^rel_i - 1) / log2(i+1)
        
        Args:
            retrieved: Ranked list of retrieved results
            k: Cutoff rank
            use_graded: Use graded relevance (True) or binary (False)
            
        Returns:
            DCG at rank K
        """
        dcg = 0.0
        
        for i, result in enumerate(retrieved[:k]):
            rank = i + 1
            
            if use_graded:
                # Graded relevance: (2^rel - 1) / log2(rank+1)
                rel = result.relevance
                gain = (2 ** rel - 1) / math.log2(rank + 1)
            else:
                # Binary relevance: rel / log2(rank+1)
                rel = 1.0 if result.relevance >= self.relevance_threshold else 0.0
                gain = rel / math.log2(rank + 1)
            
            dcg += gain
        
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved: List[RetrievalResult],
        relevant_ids: Set[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.
        
        NDCG@K = DCG@K / IDCG@K
        
        Where IDCG is the ideal DCG (perfect ranking).
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs
            k: Cutoff rank
            relevance_scores: Optional graded relevance scores
            
        Returns:
            NDCG at rank K (0 to 1)
        """
        if not relevant_ids:
            return 0.0
        
        # Compute actual DCG
        dcg = self.dcg_at_k(retrieved, k, use_graded=relevance_scores is not None)
        
        # Compute ideal DCG
        if relevance_scores:
            # Sort by relevance scores for ideal ranking
            ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
            idcg = sum(
                (2 ** rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(ideal_rels)
            )
        else:
            # Binary relevance: ideal is all relevant docs at top
            n_relevant = len(relevant_ids)
            ideal_k = min(k, n_relevant)
            idcg = sum(
                1.0 / math.log2(i + 2) for i in range(ideal_k)
            )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mrr(self, retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
        """
        Compute Mean Reciprocal Rank for a single query.
        
        RR = 1 / rank_of_first_relevant_doc
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs
            
        Returns:
            Reciprocal rank (0 if no relevant doc found)
        """
        for i, result in enumerate(retrieved):
            if result.doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def hit_rate_at_k(
        self,
        retrieved: List[RetrievalResult],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Compute Hit Rate at K (binary: 1 if any relevant doc in top-K, else 0).
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs
            k: Cutoff rank
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = retrieved[:k]
        
        for result in top_k:
            if result.doc_id in relevant_ids:
                return 1.0
        
        return 0.0
    
    def average_precision(
        self,
        retrieved: List[RetrievalResult],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Average Precision for a single query.
        
        AP = (1/|R|) Σ P@k * rel_k
        
        Where k ranges over ranks of relevant documents.
        
        Args:
            retrieved: Ranked list of retrieved results
            relevant_ids: Set of relevant document IDs
            
        Returns:
            Average precision
        """
        if not relevant_ids:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(retrieved):
            if result.doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_ids)
    
    def compute_all_metrics(
        self,
        query_results: List[QueryResult],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> BenchmarkMetrics:
        """
        Compute all retrieval metrics across multiple queries.
        
        Args:
            query_results: List of query results
            k_values: List of K values to compute metrics at
            
        Returns:
            Aggregated benchmark metrics
        """
        metrics = BenchmarkMetrics()
        
        n_queries = len(query_results)
        if n_queries == 0:
            return metrics
        
        # Initialize accumulators
        precision_sums = {k: 0.0 for k in k_values}
        recall_sums = {k: 0.0 for k in k_values}
        ndcg_sums = {k: 0.0 for k in k_values}
        hit_sums = {k: 0.0 for k in k_values}
        f1_sums = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        ap_sum = 0.0
        
        # Process each query
        for qr in query_results:
            query_metrics: Dict[str, float] = {}
            
            # Compute per-query metrics
            for k in k_values:
                p_k = self.precision_at_k(qr.retrieved, qr.relevant_doc_ids, k)
                r_k = self.recall_at_k(qr.retrieved, qr.relevant_doc_ids, k)
                n_k = self.ndcg_at_k(qr.retrieved, qr.relevant_doc_ids, k)
                h_k = self.hit_rate_at_k(qr.retrieved, qr.relevant_doc_ids, k)
                f_k = self.f1_at_k(qr.retrieved, qr.relevant_doc_ids, k)
                
                precision_sums[k] += p_k
                recall_sums[k] += r_k
                ndcg_sums[k] += n_k
                hit_sums[k] += h_k
                f1_sums[k] += f_k
                
                query_metrics[f"P@{k}"] = p_k
                query_metrics[f"R@{k}"] = r_k
                query_metrics[f"NDCG@{k}"] = n_k
                query_metrics[f"HR@{k}"] = h_k
                query_metrics[f"F1@{k}"] = f_k
            
            rr = self.mrr(qr.retrieved, qr.relevant_doc_ids)
            ap = self.average_precision(qr.retrieved, qr.relevant_doc_ids)
            
            mrr_sum += rr
            ap_sum += ap
            
            query_metrics["MRR"] = rr
            query_metrics["AP"] = ap
            
            metrics.per_query_metrics[qr.query_id] = query_metrics
        
        # Compute averages
        for k in k_values:
            metrics.precision_at_k[k] = precision_sums[k] / n_queries
            metrics.recall_at_k[k] = recall_sums[k] / n_queries
            metrics.ndcg_at_k[k] = ndcg_sums[k] / n_queries
            metrics.hit_rate_at_k[k] = hit_sums[k] / n_queries
            metrics.f1_at_k[k] = f1_sums[k] / n_queries
        
        metrics.mrr = mrr_sum / n_queries
        metrics.map_score = ap_sum / n_queries
        
        return metrics
    
    def compare_systems(
        self,
        system_results: Dict[str, List[QueryResult]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Compare multiple retrieval systems.
        
        Args:
            system_results: Dict mapping system name to query results
            k_values: K values to compute metrics at
            
        Returns:
            Dict mapping system name to benchmark metrics
        """
        return {
            system_name: self.compute_all_metrics(results, k_values)
            for system_name, results in system_results.items()
        }


def create_query_result(
    query_id: str,
    query_text: str,
    retrieved_ids: List[str],
    retrieved_scores: List[float],
    relevant_ids: List[str],
    relevance_scores: Optional[List[float]] = None
) -> QueryResult:
    """
    Factory function to create QueryResult from simple inputs.
    
    Args:
        query_id: Query identifier
        query_text: Query text
        retrieved_ids: List of retrieved document IDs (ranked)
        retrieved_scores: List of retrieval scores
        relevant_ids: List of relevant document IDs
        relevance_scores: Optional graded relevance scores
        
    Returns:
        QueryResult object
    """
    # Create retrieval results
    rel_scores_map = {}
    if relevance_scores:
        for doc_id, score in zip(relevant_ids, relevance_scores):
            rel_scores_map[doc_id] = score
    
    retrieved = []
    for i, (doc_id, score) in enumerate(zip(retrieved_ids, retrieved_scores)):
        relevance = rel_scores_map.get(doc_id, 1.0 if doc_id in relevant_ids else 0.0)
        retrieved.append(RetrievalResult(
            doc_id=doc_id,
            rank=i + 1,
            score=score,
            relevance=relevance
        ))
    
    return QueryResult(
        query_id=query_id,
        query_text=query_text,
        retrieved=retrieved,
        relevant_doc_ids=set(relevant_ids)
    )
