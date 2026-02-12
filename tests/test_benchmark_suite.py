"""
Tests for TreeRAG Benchmark Suite.

Tests all metrics modules:
- Retrieval metrics (P@K, R@K, NDCG, MRR, MAP)
- Efficiency metrics (latency, tokens, traversal)
- Fidelity metrics (groundedness, hallucination)
- Statistical tests (t-test, Wilcoxon, bootstrap)
"""

import pytest
import math
import json
from pathlib import Path

from benchmarks.metrics.retrieval_metrics import (
    RetrievalMetrics,
    QueryResult,
    RetrievalResult,
    BenchmarkMetrics,
    create_query_result
)
from benchmarks.metrics.efficiency_metrics import (
    EfficiencyMetrics,
    LatencyMeasurement,
    TokenUsage,
    TraversalStats,
    LatencyTimer,
    compare_token_efficiency
)
from benchmarks.metrics.fidelity_metrics import (
    FidelityMetrics,
    FidelityAnalysis,
    Claim,
    HallucinationType,
    CitationAccuracy
)
from benchmarks.metrics.statistical_tests import (
    StatisticalTests,
    StatisticalTestResult,
    ComparisonSummary,
    TestType,
    generate_latex_table
)


class TestRetrievalMetrics:
    """Tests for retrieval metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics calculator."""
        return RetrievalMetrics()
    
    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(doc_id="d1", rank=1, score=0.9, relevance=1.0),
            RetrievalResult(doc_id="d2", rank=2, score=0.8, relevance=0.0),
            RetrievalResult(doc_id="d3", rank=3, score=0.7, relevance=1.0),
            RetrievalResult(doc_id="d4", rank=4, score=0.6, relevance=0.0),
            RetrievalResult(doc_id="d5", rank=5, score=0.5, relevance=1.0),
        ]
    
    @pytest.fixture
    def relevant_ids(self):
        """Create relevant document IDs."""
        return {"d1", "d3", "d5", "d7"}  # 4 relevant docs, 3 in top-5
    
    def test_precision_at_k(self, metrics, sample_results, relevant_ids):
        """Test Precision@K calculation."""
        # P@5 = 3/5 = 0.6 (3 relevant in top 5)
        p5 = metrics.precision_at_k(sample_results, relevant_ids, k=5)
        assert p5 == 0.6
        
        # P@1 = 1/1 = 1.0 (d1 is relevant)
        p1 = metrics.precision_at_k(sample_results, relevant_ids, k=1)
        assert p1 == 1.0
        
        # P@2 = 1/2 = 0.5 (only d1 is relevant)
        p2 = metrics.precision_at_k(sample_results, relevant_ids, k=2)
        assert p2 == 0.5
    
    def test_recall_at_k(self, metrics, sample_results, relevant_ids):
        """Test Recall@K calculation."""
        # R@5 = 3/4 = 0.75 (3 of 4 relevant docs in top 5)
        r5 = metrics.recall_at_k(sample_results, relevant_ids, k=5)
        assert r5 == 0.75
        
        # R@1 = 1/4 = 0.25
        r1 = metrics.recall_at_k(sample_results, relevant_ids, k=1)
        assert r1 == 0.25
    
    def test_f1_at_k(self, metrics, sample_results, relevant_ids):
        """Test F1@K calculation."""
        f5 = metrics.f1_at_k(sample_results, relevant_ids, k=5)
        
        p5 = 0.6
        r5 = 0.75
        expected = 2 * p5 * r5 / (p5 + r5)
        
        assert abs(f5 - expected) < 0.001
    
    def test_ndcg_at_k(self, metrics, sample_results, relevant_ids):
        """Test NDCG@K calculation."""
        ndcg5 = metrics.ndcg_at_k(sample_results, relevant_ids, k=5)
        
        # NDCG should be between 0 and 1
        assert 0 <= ndcg5 <= 1
        
        # With some relevant docs, NDCG should be > 0
        assert ndcg5 > 0
    
    def test_mrr(self, metrics, sample_results, relevant_ids):
        """Test Mean Reciprocal Rank calculation."""
        rr = metrics.mrr(sample_results, relevant_ids)
        
        # First relevant doc is at rank 1, so RR = 1/1 = 1.0
        assert rr == 1.0
        
        # Test with non-relevant first doc
        results_delayed = [
            RetrievalResult(doc_id="d_irrelevant", rank=1, score=0.9, relevance=0.0),
            RetrievalResult(doc_id="d1", rank=2, score=0.8, relevance=1.0),
        ]
        rr_delayed = metrics.mrr(results_delayed, relevant_ids)
        assert rr_delayed == 0.5  # 1/2
    
    def test_average_precision(self, metrics, sample_results, relevant_ids):
        """Test Average Precision calculation."""
        ap = metrics.average_precision(sample_results, relevant_ids)
        
        # AP should be between 0 and 1
        assert 0 <= ap <= 1
        
        # With relevant docs, AP > 0
        assert ap > 0
    
    def test_compute_all_metrics(self, metrics):
        """Test computing all metrics across queries."""
        query_results = [
            create_query_result(
                query_id="q1",
                query_text="test query 1",
                retrieved_ids=["d1", "d2", "d3"],
                retrieved_scores=[0.9, 0.8, 0.7],
                relevant_ids=["d1", "d3"]
            ),
            create_query_result(
                query_id="q2",
                query_text="test query 2",
                retrieved_ids=["d4", "d5", "d6"],
                retrieved_scores=[0.8, 0.7, 0.6],
                relevant_ids=["d4"]
            )
        ]
        
        result = metrics.compute_all_metrics(query_results, k_values=[1, 3])
        
        assert isinstance(result, BenchmarkMetrics)
        assert 1 in result.precision_at_k
        assert 3 in result.precision_at_k
        assert result.mrr > 0
        assert len(result.per_query_metrics) == 2
    
    def test_empty_results(self, metrics):
        """Test handling of empty results."""
        p = metrics.precision_at_k([], {"d1"}, k=5)
        assert p == 0.0
        
        r = metrics.recall_at_k([], {"d1"}, k=5)
        assert r == 0.0
        
        mrr = metrics.mrr([], {"d1"})
        assert mrr == 0.0


class TestEfficiencyMetrics:
    """Tests for efficiency metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create efficiency metrics calculator."""
        return EfficiencyMetrics()
    
    def test_latency_measurement(self, metrics):
        """Test latency measurement recording."""
        measurement = LatencyMeasurement(
            query_id="q1",
            total_ms=100.0,
            llm_ms=80.0,
            traversal_ms=15.0,
            embedding_ms=5.0
        )
        
        assert measurement.overhead_ms == 20.0  # 100 - 80
        
        metrics.record_latency(measurement)
        stats = metrics.compute_latency_stats()
        
        assert stats["mean_ms"] == 100.0
    
    def test_token_usage(self, metrics):
        """Test token usage tracking."""
        usage = TokenUsage(
            query_id="q1",
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            context_tokens=800,
            original_document_tokens=5000
        )
        
        # Reduction rate = 1 - 800/5000 = 0.84
        assert abs(usage.reduction_rate - 0.84) < 0.001
        
        # Compression ratio = 5000/800 = 6.25
        assert abs(usage.compression_ratio - 6.25) < 0.001
        
        metrics.record_tokens(usage)
        stats = metrics.compute_token_stats()
        
        assert stats["reduction_mean"] == usage.reduction_rate
    
    def test_traversal_stats(self, metrics):
        """Test traversal statistics."""
        stats = TraversalStats(
            query_id="q1",
            nodes_visited=10,
            nodes_pruned=40,
            max_depth_reached=3,
            tree_total_nodes=50,
            tree_max_depth=5,
            beam_width_used=3
        )
        
        # Visit rate = 10/50 = 0.2
        assert stats.visit_rate == 0.2
        
        # Pruning rate = 40/50 = 0.8
        assert stats.pruning_rate == 0.8
        
        metrics.record_traversal(stats)
        result = metrics.compute_traversal_stats()
        
        assert result["nodes_visited_mean"] == 10
    
    def test_latency_timer(self):
        """Test latency timer context manager."""
        import time
        
        with LatencyTimer("test") as timer:
            time.sleep(0.01)  # 10ms
        
        assert timer.latency_ms >= 10
    
    def test_token_comparison(self):
        """Test token efficiency comparison."""
        treerag_tokens = [100, 120, 90, 110]
        baseline_tokens = [200, 220, 180, 210]
        
        result = compare_token_efficiency(treerag_tokens, baseline_tokens)
        
        assert result["treerag_total"] == 420
        assert result["baseline_total"] == 810
        assert result["total_reduction"] > 0.4  # ~48% reduction
    
    def test_percentile_calculation(self, metrics):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        p50 = metrics.percentile(values, 50)
        assert abs(p50 - 5.5) < 0.1
        
        p95 = metrics.percentile(values, 95)
        assert p95 >= 9


class TestFidelityMetrics:
    """Tests for fidelity metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create fidelity metrics calculator."""
        return FidelityMetrics()
    
    def test_claim_extraction(self, metrics):
        """Test claim extraction from answer."""
        answer = "The system uses 128 nodes. It was built in 2024. John Smith designed it."
        
        claims = metrics.extract_claims_simple(answer)
        
        assert len(claims) >= 3
        
        # Check claim types
        claim_types = [c.claim_type for c in claims]
        assert "number" in claim_types or "temporal" in claim_types
    
    def test_claim_verification(self, metrics):
        """Test claim verification against context."""
        claim = Claim(
            text="The system uses 128 nodes for processing.",
            claim_type="number",
            grounded=False
        )
        
        context_supporting = "The architecture includes 128 processing nodes."
        context_not_supporting = "The system has a modular design."
        
        grounded1, _ = metrics.verify_claim_simple(claim, context_supporting)
        grounded2, _ = metrics.verify_claim_simple(claim, context_not_supporting)
        
        assert grounded1 == True
        assert grounded2 == False
    
    def test_fidelity_analysis(self, metrics):
        """Test full fidelity analysis."""
        answer = "The revenue was $5 million. The company has 100 employees."
        context = "Annual revenue reached $5 million last year. The workforce consists of 100 full-time employees."
        
        analysis = metrics.analyze_answer("q1", answer, context)
        
        assert analysis.query_id == "q1"
        assert analysis.total_claims >= 2
        assert analysis.groundedness_score > 0
    
    def test_hallucination_classification(self, metrics):
        """Test hallucination type classification."""
        # Number error
        claim = Claim(text="Revenue was $10 million.", claim_type="number", grounded=False)
        h_type = metrics._classify_hallucination(claim, "Revenue was $5 million.")
        assert h_type == HallucinationType.QUANTITATIVE_ERROR
        
        # Entity error
        claim = Claim(text="John Smith led the project.", claim_type="entity", grounded=False)
        h_type = metrics._classify_hallucination(claim, "The project was completed.")
        assert h_type == HallucinationType.FABRICATED_ENTITY
    
    def test_compute_metrics(self, metrics):
        """Test aggregated metrics computation."""
        analyses = [
            FidelityAnalysis(
                query_id="q1",
                answer_text="Test answer 1.",
                context_text="Test context 1.",
                claims=[
                    Claim(text="Claim 1", claim_type="fact", grounded=True),
                    Claim(text="Claim 2", claim_type="fact", grounded=True),
                ]
            ),
            FidelityAnalysis(
                query_id="q2",
                answer_text="Test answer 2.",
                context_text="Test context 2.",
                claims=[
                    Claim(text="Claim 3", claim_type="fact", grounded=True),
                    Claim(text="Claim 4", claim_type="fact", grounded=False),
                ]
            )
        ]
        
        result = metrics.compute_metrics(analyses)
        
        assert result.total_claims == 4
        assert result.total_grounded == 3
        assert result.total_hallucinated == 1
        assert result.groundedness_mean == 0.75  # (1.0 + 0.5) / 2


class TestCitationAccuracy:
    """Tests for citation accuracy."""
    
    @pytest.fixture
    def checker(self):
        """Create citation accuracy checker."""
        return CitationAccuracy()
    
    def test_extract_citations(self, checker):
        """Test citation extraction."""
        answer = "According to the report [1], revenue increased. This aligns with [2] findings."
        
        citations = checker.extract_citations(answer)
        
        assert len(citations) == 2
        assert citations[0][0] == "1"
        assert citations[1][0] == "2"
    
    def test_verify_citations(self, checker):
        """Test citation verification."""
        answer = "Revenue grew by 15% according to the report [1]. Operating costs were reduced through automation [2]."
        sources = {
            "1": "The company reported 15% revenue growth in Q3.",
            "2": "Operating costs were reduced through automation."
        }
        
        result = checker.verify_citations(answer, sources)
        
        assert result["total_citations"] == 2
        # At least one citation should be verified
        assert result["verified"] >= 0


class TestStatisticalTests:
    """Tests for statistical tests."""
    
    @pytest.fixture
    def stats(self):
        """Create statistical tests calculator."""
        return StatisticalTests(alpha=0.05, random_seed=42)
    
    def test_paired_ttest(self, stats):
        """Test paired t-test."""
        # Clear difference between groups
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.89]
        scores_b = [0.7, 0.65, 0.68, 0.72, 0.67, 0.70, 0.66, 0.69]
        
        result = stats.paired_ttest(scores_a, scores_b)
        
        assert result.test_type == TestType.PAIRED_TTEST
        assert result.statistic > 0
        assert result.significant == True
        assert result.p_value < 0.05
        assert result.effect_size > 0
    
    def test_ttest_no_difference(self, stats):
        """Test t-test with no significant difference."""
        scores_a = [0.80, 0.82, 0.79, 0.81]
        scores_b = [0.79, 0.81, 0.80, 0.82]
        
        result = stats.paired_ttest(scores_a, scores_b)
        
        # Should not be significant
        assert result.significant == False or result.effect_size < 0.2
    
    def test_wilcoxon_signed_rank(self, stats):
        """Test Wilcoxon signed-rank test."""
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87]
        scores_b = [0.7, 0.65, 0.68, 0.72, 0.67]
        
        result = stats.wilcoxon_signed_rank(scores_a, scores_b)
        
        assert result.test_type == TestType.WILCOXON
        assert result.n_samples == 5
    
    def test_bootstrap_ci(self, stats):
        """Test bootstrap confidence interval."""
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90]
        scores_b = [0.7, 0.65, 0.68, 0.72, 0.67, 0.70]
        
        result = stats.bootstrap_ci(scores_a, scores_b, n_bootstrap=1000)
        
        assert result.test_type == TestType.BOOTSTRAP
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower < result.ci_upper
        
        # CI should exclude 0 for this clear difference
        assert result.ci_lower > 0 or result.ci_upper < 0
    
    def test_permutation_test(self, stats):
        """Test permutation test."""
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87]
        scores_b = [0.7, 0.65, 0.68, 0.72, 0.67]
        
        result = stats.permutation_test(scores_a, scores_b, n_permutations=1000)
        
        assert result.test_type == TestType.PERMUTATION
        assert 0 <= result.p_value <= 1
    
    def test_cohens_d(self, stats):
        """Test Cohen's d effect size."""
        # Large effect
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87]
        scores_b = [0.5, 0.45, 0.48, 0.52, 0.47]
        
        d = stats.cohens_d(scores_a, scores_b)
        
        assert d > 0.8  # Large effect
        assert stats._interpret_cohens_d(d) == "large"
    
    def test_effect_size_interpretation(self, stats):
        """Test effect size interpretation."""
        assert stats._interpret_cohens_d(0.1) == "negligible"
        assert stats._interpret_cohens_d(0.3) == "small"
        assert stats._interpret_cohens_d(0.6) == "medium"
        assert stats._interpret_cohens_d(1.0) == "large"
    
    def test_compare_methods(self, stats):
        """Test comprehensive method comparison."""
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90]
        scores_b = [0.7, 0.65, 0.68, 0.72, 0.67, 0.70]
        
        summary = stats.compare_methods(
            "TreeRAG", "FlatRAG",
            scores_a, scores_b,
            "P@5"
        )
        
        assert summary.method_a == "TreeRAG"
        assert summary.method_b == "FlatRAG"
        assert summary.metric == "P@5"
        assert "paired_ttest" in summary.tests
        assert "wilcoxon" in summary.tests
        assert "bootstrap" in summary.tests
        assert summary.winner == "TreeRAG"
        assert summary.confidence in ["low", "medium", "high"]
    
    def test_bonferroni_correction(self, stats):
        """Test Bonferroni correction."""
        p_values = [0.005, 0.02, 0.03, 0.04, 0.05]
        
        significant, corrected_alpha = stats.bonferroni_correction(p_values)
        
        assert corrected_alpha == 0.01  # 0.05 / 5
        assert significant[0] == True   # 0.005 < 0.01
        assert significant[4] == False  # 0.05 > 0.01
    
    def test_benjamini_hochberg(self, stats):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.10]
        
        significant, adjusted = stats.benjamini_hochberg(p_values)
        
        assert len(adjusted) == 5
        assert adjusted[0] <= adjusted[1]  # Monotonicity
    
    def test_generate_latex_table(self, stats):
        """Test LaTeX table generation."""
        scores_a = [0.9, 0.85, 0.88]
        scores_b = [0.7, 0.65, 0.68]
        
        summary = stats.compare_methods(
            "TreeRAG", "FlatRAG",
            scores_a, scores_b,
            "P@5"
        )
        
        latex = generate_latex_table([summary])
        
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "P@5" in latex


class TestBenchmarkDataset:
    """Test benchmark dataset loading."""
    
    def test_load_benchmark_questions(self):
        """Test loading benchmark questions."""
        dataset_path = Path("benchmarks/datasets/benchmark_questions.json")
        
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "questions" in data
            assert len(data["questions"]) > 0
            
            # Check question structure
            q = data["questions"][0]
            assert "question_id" in q
            assert "question" in q
            assert "relevant_sections" in q
            assert "expected_answer" in q


class TestMathFunctions:
    """Test mathematical utility functions."""
    
    @pytest.fixture
    def stats(self):
        return StatisticalTests()
    
    def test_mean(self, stats):
        """Test mean calculation."""
        assert stats.mean([1, 2, 3, 4, 5]) == 3.0
        assert stats.mean([]) == 0.0
    
    def test_std(self, stats):
        """Test standard deviation calculation."""
        std = stats.std([1, 2, 3, 4, 5])
        assert abs(std - 1.5811) < 0.01
    
    def test_normal_cdf(self, stats):
        """Test normal CDF approximation."""
        # Standard normal properties
        assert abs(stats._normal_cdf(0) - 0.5) < 0.001
        assert stats._normal_cdf(3) > 0.99
        assert stats._normal_cdf(-3) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
