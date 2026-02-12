
import pytest
import math
from typing import Dict

from src.core.theoretical_analysis import (
    TreeParameters,
    TraversalStrategy,
    ComplexityBounds,
    OptimalityAnalysis,
    TokenReductionAnalysis,
    ConvergenceAnalysis,
    ComplexityAnalyzer,
    OptimalityAnalyzer,
    TokenReductionAnalyzer,
    ConvergenceAnalyzer,
    TheoreticalFramework,
    analyze_tree,
    generate_paper_appendix
)


class TestTreeParameters:
    
    def test_basic_initialization(self):
        params = TreeParameters(
            branching_factor=4,
            depth=5,
            total_nodes=100
        )
        
        assert params.branching_factor == 4
        assert params.depth == 5
        assert params.total_nodes == 100
        assert params.avg_node_tokens == 100           
    
    def test_leaf_node_estimation(self):
        params = TreeParameters(
            branching_factor=2,
            depth=3,
            total_nodes=15
        )
        
                                                     
        assert params.leaf_nodes == 8


class TestComplexityAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return ComplexityAnalyzer()
    
    @pytest.fixture
    def sample_params(self):
        return TreeParameters(
            branching_factor=4,
            depth=5,
            total_nodes=500
        )
    
    def test_greedy_complexity(self, analyzer, sample_params):
        result = analyzer.analyze(
            sample_params,
            TraversalStrategy.GREEDY
        )
        
        assert isinstance(result, ComplexityBounds)
        assert "O(b·d)" in result.time_expected
        assert result.estimated_operations == 4 * 5         
    
    def test_beam_search_complexity(self, analyzer, sample_params):
        beam_width = 3
        result = analyzer.analyze(
            sample_params,
            TraversalStrategy.BEAM_SEARCH,
            beam_width=beam_width
        )
        
                              
        assert result.estimated_operations == beam_width * 4 * 5
    
    def test_exhaustive_complexity(self, analyzer, sample_params):
        result = analyzer.analyze(
            sample_params,
            TraversalStrategy.EXHAUSTIVE
        )
        
                      
        assert result.estimated_operations == 500
    
    def test_speedup_calculation(self, analyzer, sample_params):
        result = analyzer.analyze(
            sample_params,
            TraversalStrategy.GREEDY
        )
        
                                               
        assert result.vs_flat_speedup == 500 / 20
    
    def test_memory_estimation(self, analyzer, sample_params):
        result = analyzer.analyze(
            sample_params,
            TraversalStrategy.GREEDY
        )
        
                                              
        assert result.estimated_memory_mb > 0
    
    def test_to_dict(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        d = result.to_dict()
        
        assert "time_complexity" in d
        assert "space_complexity" in d
        assert "speedup" in d
    
    def test_derive_bounds_proof(self, analyzer, sample_params):
        proof = analyzer.derive_bounds_proof(sample_params)
        
        assert "\\begin{theorem}" in proof
        assert "\\begin{proof}" in proof
        assert "O(" in proof


class TestOptimalityAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return OptimalityAnalyzer()
    
    @pytest.fixture
    def sample_params(self):
        return TreeParameters(
            branching_factor=4,
            depth=5,
            total_nodes=500
        )
    
    def test_greedy_analysis(self, analyzer, sample_params):
        result = analyzer.analyze_greedy(sample_params)
        
        assert isinstance(result, OptimalityAnalysis)
        assert result.strategy == TraversalStrategy.GREEDY
        assert not result.is_optimal                                
        assert 0 < result.approximation_ratio < 1
    
    def test_greedy_approximation_ratio(self, analyzer, sample_params):
        result = analyzer.analyze_greedy(sample_params)
        
        expected = 1 - 1/math.e           
        assert abs(result.approximation_ratio - expected) < 0.01
    
    def test_beam_search_analysis(self, analyzer, sample_params):
        result = analyzer.analyze_beam_search(sample_params, beam_width=5)
        
        assert result.strategy == TraversalStrategy.BEAM_SEARCH
                                                                  
        greedy_result = analyzer.analyze_greedy(sample_params)
        assert result.approximation_ratio >= greedy_result.approximation_ratio
    
    def test_optimality_conditions(self, analyzer, sample_params):
        result = analyzer.analyze_greedy(sample_params)
        
        assert len(result.optimality_conditions) > 0
        assert len(result.failure_cases) > 0
    
    def test_time_quality_tradeoff(self, analyzer, sample_params):
        result = analyzer.analyze_greedy(sample_params)
        
                                                        
        tradeoff = result.time_vs_quality
        
                                          
        items = sorted(
            [(float(k.replace("x", "")), v) for k, v in tradeoff.items()],
            key=lambda x: x[0]
        )
        
                                                    
        for i in range(1, len(items)):
            assert items[i][1] >= items[i-1][1]
    
    def test_to_dict(self, analyzer, sample_params):
        result = analyzer.analyze_greedy(sample_params)
        d = result.to_dict()
        
        assert "strategy" in d
        assert "approximation_ratio" in d
        assert "optimality_conditions" in d


class TestTokenReductionAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return TokenReductionAnalyzer()
    
    @pytest.fixture
    def sample_params(self):
        return TreeParameters(
            branching_factor=4,
            depth=5,
            total_nodes=500,
            avg_node_tokens=100
        )
    
    def test_reduction_analysis(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        
        assert isinstance(result, TokenReductionAnalysis)
        assert 0 <= result.expected_reduction <= 1
    
    def test_significant_reduction(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        
                                                  
        assert result.expected_reduction > 0.5
    
    def test_tree_vs_flat_comparison(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        
                                         
        assert result.tree_rag_tokens <= result.flat_rag_tokens
    
    def test_bounds_valid(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        
        lower, upper = result.reduction_bounds
        assert 0 <= lower <= upper <= 1
        assert lower <= result.expected_reduction <= upper
    
    def test_compression_ratio(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        
                                                            
        assert result.compression_ratio >= 1.0
    
    def test_to_dict(self, analyzer, sample_params):
        result = analyzer.analyze(sample_params)
        d = result.to_dict()
        
        assert "reduction_ratio" in d
        assert "bounds" in d
        assert "token_comparison" in d


class TestConvergenceAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return ConvergenceAnalyzer()
    
    def test_basic_analysis(self, analyzer):
        result = analyzer.analyze(
            feature_dim=8,
            learning_rate=0.01,
            target_error=0.01
        )
        
        assert isinstance(result, ConvergenceAnalysis)
        assert result.learning_rate == 0.01
        assert result.required_samples > 0
    
    def test_sample_complexity_scaling(self, analyzer):
        result1 = analyzer.analyze(target_error=0.1)
        result2 = analyzer.analyze(target_error=0.01)
        
                                                        
        ratio = result2.required_samples / result1.required_samples
        assert ratio > 50                        
    
    def test_convergence_rate(self, analyzer):
        result = analyzer.analyze()
        
        assert "√T" in result.convergence_rate or "sqrt" in result.convergence_rate.lower()
    
    def test_stability_condition(self, analyzer):
        result = analyzer.analyze(learning_rate=0.1)
        
        assert "η" in result.stability_condition or "≤" in result.stability_condition
    
    def test_to_dict(self, analyzer):
        result = analyzer.analyze()
        d = result.to_dict()
        
        assert "learning_rate" in d
        assert "convergence_rate" in d
        assert "required_samples" in d


class TestTheoreticalFramework:
    
    @pytest.fixture
    def framework(self):
        return TheoreticalFramework()
    
    @pytest.fixture
    def sample_params(self):
        return TreeParameters(
            branching_factor=4,
            depth=5,
            total_nodes=500
        )
    
    def test_full_analysis(self, framework, sample_params):
        result = framework.full_analysis(sample_params)
        
        assert "parameters" in result
        assert "complexity" in result
        assert "optimality" in result
        assert "token_reduction" in result
        assert "convergence" in result
        assert "summary" in result
    
    def test_analysis_with_beam_search(self, framework, sample_params):
        result = framework.full_analysis(
            sample_params,
            strategy=TraversalStrategy.BEAM_SEARCH,
            beam_width=5
        )
        
        assert result["parameters"]["strategy"] == "beam_search"
        assert "beam_search" in str(result["optimality"]["strategy"])
    
    def test_summary_generation(self, framework, sample_params):
        result = framework.full_analysis(sample_params)
        summary = result["summary"]
        
        assert "time_complexity" in summary
        assert "speedup_vs_flat" in summary
        assert "token_reduction" in summary
        assert "key_findings" in summary
        assert len(summary["key_findings"]) > 0
    
    def test_latex_appendix_generation(self, framework, sample_params):
        latex = framework.generate_latex_appendix(sample_params)
        
                                       
        assert "\\appendix" in latex
        assert "\\section" in latex
        assert "\\begin{theorem}" in latex
        assert "\\begin{proof}" in latex
    
    def test_latex_completeness(self, framework, sample_params):
        latex = framework.generate_latex_appendix(sample_params)
        
                                                                    
        assert "Time Complexity" in latex
        assert "Optimality" in latex
        assert "Token Reduction" in latex
        assert "Convergence" in latex


class TestConvenienceFunctions:
    
    def test_analyze_tree(self):
        result = analyze_tree(
            branching_factor=4,
            depth=5,
            total_nodes=500,
            strategy="greedy"
        )
        
        assert isinstance(result, dict)
        assert "complexity" in result
        assert "summary" in result
    
    def test_analyze_tree_beam_search(self):
        result = analyze_tree(
            branching_factor=4,
            depth=5,
            total_nodes=500,
            strategy="beam_search"
        )
        
        assert result["parameters"]["strategy"] == "beam_search"
    
    def test_generate_paper_appendix(self):
        latex = generate_paper_appendix(
            branching_factor=4,
            depth=5,
            total_nodes=500
        )
        
        assert isinstance(latex, str)
        assert "\\appendix" in latex
        assert len(latex) > 1000                       


class TestEdgeCases:
    
    def test_small_tree(self):
        params = TreeParameters(
            branching_factor=2,
            depth=1,
            total_nodes=3
        )
        
        framework = TheoreticalFramework()
        result = framework.full_analysis(params)
        
                                   
        assert result["complexity"]["estimates"]["operations"] == 2         
    
    def test_large_tree(self):
        params = TreeParameters(
            branching_factor=10,
            depth=10,
            total_nodes=10000
        )
        
        framework = TheoreticalFramework()
        result = framework.full_analysis(params)
        
                                       
        speedup = float(result["summary"]["speedup_vs_flat"].replace("x", ""))
        assert speedup > 50
    
    def test_deep_narrow_tree(self):
        params = TreeParameters(
            branching_factor=2,
            depth=20,
            total_nodes=100
        )
        
        framework = TheoreticalFramework()
        result = framework.full_analysis(params)
        
        assert result is not None
    
    def test_wide_shallow_tree(self):
        params = TreeParameters(
            branching_factor=50,
            depth=2,
            total_nodes=100
        )
        
        framework = TheoreticalFramework()
        result = framework.full_analysis(params)
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
