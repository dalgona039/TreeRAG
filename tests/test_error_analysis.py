
import pytest
import math
from typing import List

from src.core.error_analysis import (
    ErrorAnalyzer,
    ConfidenceCalibrator,
    HallucinationQuantifier,
    ErrorInstance,
    ErrorCategory,
    SeverityLevel,
    CalibrationBin,
    CalibrationResult,
    ErrorAnalysisResult,
    generate_error_report
)


class TestErrorAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return ErrorAnalyzer()
    
    def test_classify_retrieval_miss(self, analyzer):
        errors = analyzer.classify_error(
            query_id="q1",
            answer="The revenue was $5 million.",
            reference="The revenue was $5 million with 15% growth.",
            context="Unrelated content about weather.",
            confidence=0.8
        )
        
                                      
        assert any(e.category == ErrorCategory.RETRIEVAL_MISS for e in errors)
    
    def test_classify_hallucination(self, analyzer):
        errors = analyzer.classify_error(
            query_id="q1",
            answer="John Smith reported $10 million in revenue.",
            reference="The revenue was reported.",
            context="The revenue was reported as growing.",
            confidence=0.9
        )
        
                                                                 
        assert any(e.category == ErrorCategory.GENERATION_HALLUCINATION for e in errors)
    
    def test_classify_overclaiming(self, analyzer):
        errors = analyzer.classify_error(
            query_id="q1",
            answer="This approach definitely and absolutely works every time.",
            reference="The approach works well.",
            context="The approach works well in most cases.",
            confidence=0.95
        )
        
                                    
        assert any(e.category == ErrorCategory.REASONING_OVERCLAIMED for e in errors)
    
    def test_no_errors(self, analyzer):
        errors = analyzer.classify_error(
            query_id="q1",
            answer="The revenue was $5 million.",
            reference="The revenue was $5 million.",
            context="The company reported revenue of $5 million for the quarter.",
            confidence=0.9
        )
        
                                           
        critical_errors = [e for e in errors if e.severity == SeverityLevel.CRITICAL]
        assert len(critical_errors) == 0
    
    def test_analyze_aggregated(self, analyzer):
                         
        analyzer.classify_error(
            query_id="q1",
            answer="Answer with error.",
            reference="Expected answer.",
            context="Unrelated context.",
            confidence=0.8
        )
        
        analyzer.classify_error(
            query_id="q2",
            answer="Answer with John Smith mentioned.",
            reference="Expected answer.",
            context="Some context without names.",
            confidence=0.7
        )
        
        result = analyzer.analyze(queries_with_errors=2, total_queries=5)
        
        assert isinstance(result, ErrorAnalysisResult)
        assert result.total_queries == 5
        assert result.queries_with_errors == 2
        assert result.error_rate == 0.4
        assert len(result.errors) > 0
    
    def test_error_instance_to_dict(self):
        error = ErrorInstance(
            error_id="err_001",
            category=ErrorCategory.GENERATION_HALLUCINATION,
            severity=SeverityLevel.MAJOR,
            query_id="q1",
            description="Test error",
            evidence="Test evidence",
            expected="Expected value",
            confidence=0.8
        )
        
        d = error.to_dict()
        
        assert d["error_id"] == "err_001"
        assert d["category"] == "generation_hallucination"
        assert d["severity"] == "major"


class TestConfidenceCalibrator:
    
    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator(n_bins=10)
    
    def test_perfect_calibration(self, calibrator):
                                          
        confidences = []
        correctness = []
        
        for conf in [0.2, 0.4, 0.6, 0.8]:
            n_samples = 100
            n_correct = int(conf * n_samples)
            
            for _ in range(n_samples):
                confidences.append(conf)
            
            correctness.extend([True] * n_correct + [False] * (n_samples - n_correct))
        
        result = calibrator.analyze(confidences, correctness)
        
                                                           
        assert result.expected_calibration_error < 0.15
    
    def test_overconfident_calibration(self, calibrator):
                                       
        confidences = [0.9] * 100
        correctness = [i < 50 for i in range(100)]                
        
        result = calibrator.analyze(confidences, correctness)
        
                                                               
        assert result.average_confidence > result.overall_accuracy + 0.3
    
    def test_brier_score(self, calibrator):
        confidences = [0.8, 0.8, 0.2, 0.2]
        correctness = [True, False, False, True]
        
        result = calibrator.analyze(confidences, correctness)
        
                                                                              
                                               
        assert abs(result.brier_score - 0.34) < 0.01
    
    def test_calibration_bins(self, calibrator):
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        correctness = [False, False, True, True, True, True, True, True, True]
        
        result = calibrator.analyze(confidences, correctness)
        
                                       
        non_empty_bins = [b for b in result.bins if b.n_samples > 0]
        assert len(non_empty_bins) > 0
        
                                                            
        for bin in result.bins:
            assert 0 <= bin.mean_confidence <= 1
            assert 0 <= bin.accuracy <= 1
    
    def test_temperature_scaling(self, calibrator):
                                   
        confidences = [0.95] * 50 + [0.05] * 50
        correctness = [i < 25 for i in range(50)] + [i >= 25 for i in range(50)]
        
        temp, calibrated = calibrator.calibrate_temperature(confidences, correctness)
        
        assert temp > 0                                  
        assert len(calibrated) == len(confidences)
        
                                                       
        assert all(0 < c < 1 for c in calibrated)
    
    def test_reliability_diagram_data(self, calibrator):
        confidences = [0.3, 0.5, 0.7, 0.9]
        correctness = [False, True, True, True]
        
        result = calibrator.analyze(confidences, correctness)
        diagram_data = calibrator.reliability_diagram_data(result)
        
        assert "bins" in diagram_data
        assert "ece" in diagram_data
        assert "mce" in diagram_data
    
    def test_empty_input(self, calibrator):
        result = calibrator.analyze([], [])
        
        assert result.expected_calibration_error == 0
        assert len(result.bins) == 0


class TestHallucinationQuantifier:
    
    @pytest.fixture
    def quantifier(self):
        return HallucinationQuantifier()
    
    def test_detect_hallucination(self, quantifier):
        result = quantifier.detect(
            query_id="q1",
            answer="John Smith reported $10 million in revenue last year.",
            context="The company reported revenue growth.",
            confidence=0.9
        )
        
        assert "hallucinations" in result
        assert "hallucination_score" in result
        assert result["hallucination_score"] >= 0
    
    def test_no_hallucination(self, quantifier):
        result = quantifier.detect(
            query_id="q1",
            answer="The company reported revenue growth this quarter.",
            context="The company reported revenue growth this quarter.",
            confidence=0.8
        )
        
                                                             
        assert result["hallucination_score"] <= 0.5
    
    def test_compute_metrics(self, quantifier):
                             
        quantifier.detect(
            query_id="q1",
            answer="This is a supported claim.",
            context="This is a supported claim in the context.",
            confidence=0.9
        )
        
        quantifier.detect(
            query_id="q2",
            answer="John Smith made an unsupported claim about Mars.",
            context="The document discusses Earth.",
            confidence=0.7
        )
        
        metrics = quantifier.compute_metrics()
        
        assert "total_queries" in metrics
        assert "hallucination_rate" in metrics
        assert "avg_hallucination_score" in metrics
        assert "confidence_hallucination_correlation" in metrics
        
        assert metrics["total_queries"] == 2
    
    def test_severity_assessment(self, quantifier):
                                        
        sev1 = quantifier._assess_severity("Revenue was $10 million.")
        assert sev1 == "high"
        
                                             
        sev2 = quantifier._assess_severity("John Smith said this.")
        assert sev2 == "high"
        
                                             
        sev3 = quantifier._assess_severity("This always happens.")
        assert sev3 == "medium"
        
                                         
        sev4 = quantifier._assess_severity("The process works well.")
        assert sev4 == "low"
    
    def test_claim_extraction(self, quantifier):
        text = "First claim here. Second claim follows. Third claim also."
        
        claims = quantifier._extract_claims(text)
        
        assert len(claims) >= 2                                  


class TestCalibrationBin:
    
    def test_gap_computation(self):
        bin = CalibrationBin(
            bin_start=0.8,
            bin_end=0.9,
            n_samples=100,
            mean_confidence=0.85,
            accuracy=0.70
        )
        
                                    
        assert abs(bin.gap - 0.15) < 0.001


class TestIntegration:
    
    def test_full_analysis_pipeline(self):
                        
        error_analyzer = ErrorAnalyzer()
        
        for i in range(5):
            error_analyzer.classify_error(
                query_id=f"q{i}",
                answer=f"Answer with John Smith number {i * 100}.",
                reference="Expected answer.",
                context="Context without specific names.",
                confidence=0.8 - i * 0.1
            )
        
        error_result = error_analyzer.analyze(
            queries_with_errors=3,
            total_queries=5
        )
        
                              
        calibrator = ConfidenceCalibrator()
        confidences = [0.8 - i * 0.1 for i in range(5)]
        correctness = [True, True, False, True, False]
        
        calibration_result = calibrator.analyze(confidences, correctness)
        
                                
        quantifier = HallucinationQuantifier()
        
        for i in range(5):
            quantifier.detect(
                query_id=f"q{i}",
                answer=f"Answer {i}",
                context="Some context",
                confidence=0.8 - i * 0.1
            )
        
        hallucination_metrics = quantifier.compute_metrics()
        
                         
        report = generate_error_report(
            error_result,
            calibration_result,
            hallucination_metrics
        )
        
        assert "summary" in report
        assert "error_analysis" in report
        assert "calibration" in report
        assert "hallucination" in report
        assert "recommendations" in report
        
                       
        assert "total_queries" in report["summary"]
        assert "error_rate" in report["summary"]
        assert "calibration_error" in report["summary"]
        
                                     
        assert isinstance(report["recommendations"], list)


class TestCorrelationComputation:
    
    def test_positive_correlation(self):
        quantifier = HallucinationQuantifier()
        
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        
        corr = quantifier._compute_correlation(x, y)
        
        assert corr > 0.5                               
    
    def test_negative_correlation(self):
        quantifier = HallucinationQuantifier()
        
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        
        corr = quantifier._compute_correlation(x, y)
        
        assert corr < -0.9                               
    
    def test_no_correlation(self):
        quantifier = HallucinationQuantifier()
        
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        
        corr = quantifier._compute_correlation(x, y)
        
        assert abs(corr) < 0.7                    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
