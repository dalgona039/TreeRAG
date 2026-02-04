"""
Unit tests for Hallucination Detector.
"""
import pytest
from src.utils.hallucination_detector import HallucinationDetector, create_detector


class TestHallucinationDetector:
    
    def test_detector_initialization(self):
        detector = HallucinationDetector(confidence_threshold=0.7)
        assert detector.confidence_threshold == 0.7
    
    def test_factory_function(self):
        detector = create_detector(confidence_threshold=0.8)
        assert isinstance(detector, HallucinationDetector)
        assert detector.confidence_threshold == 0.8
    
    def test_perfect_grounding(self):
        detector = create_detector()
        
        source_nodes = [
            {"content": "TreeRAG is a hierarchical retrieval system for documents."},
            {"content": "It uses tree-based navigation for better context."}
        ]
        
        answer = "TreeRAG is a hierarchical retrieval system."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["overall_confidence"] >= 0.8
        assert result["is_reliable"] is True
        assert result["hallucinated_count"] == 0
    
    def test_complete_hallucination(self):
        detector = create_detector()
        
        source_nodes = [
            {"content": "TreeRAG is a document retrieval system."}
        ]
        
        answer = "Quantum computing enables faster blockchain transactions."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["overall_confidence"] < 0.6
        assert result["is_reliable"] is False
        assert result["hallucinated_count"] > 0
    
    def test_partial_hallucination(self):
        detector = create_detector()
        
        source_nodes = [
            {"content": "TreeRAG uses hierarchical tree structures for document indexing."}
        ]
        
        answer = "TreeRAG uses hierarchical tree structures. It also supports quantum entanglement."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["total_sentences"] == 2
        assert result["hallucinated_count"] >= 1
    
    def test_sentence_splitting(self):
        detector = create_detector()
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = detector._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_confidence_calculation_exact_match(self):
        detector = create_detector()
        
        sentence = "This is a test sentence."
        source = "Here is some context. This is a test sentence. And more text."
        
        confidence = detector._calculate_sentence_confidence(sentence, source)
        
        assert confidence >= 0.9  
    def test_confidence_calculation_no_match(self):
        detector = create_detector()
        
        sentence = "Quantum blockchain AI revolution."
        source = "TreeRAG is a document retrieval system."
        
        confidence = detector._calculate_sentence_confidence(sentence, source)
        
        assert confidence < 0.5  
    def test_warning_formatting(self):
        detector = create_detector()
        
        source_nodes = [{"content": "Known fact A."}]
        answer = "Known fact A. Unknown claim B."
        
        result = detector.detect(answer, source_nodes)
        formatted = detector.format_with_warnings(answer, result)
        
        if result["hallucinated_count"] > 0:
            assert "⚠️" in formatted
    
    def test_summary_generation_high_confidence(self):
        detector = create_detector()
        
        result = {
            "overall_confidence": 0.95,
            "is_reliable": True,
            "hallucinated_count": 0,
            "total_sentences": 5
        }
        
        summary = detector.get_summary(result)
        
        assert "95" in summary or "95.0%" in summary
        assert "✅" in summary or "매우 높음" in summary
        assert "0" in summary  
    
    def test_summary_generation_low_confidence(self):
        detector = create_detector()
        
        result = {
            "overall_confidence": 0.40,
            "is_reliable": False,
            "hallucinated_count": 3,
            "total_sentences": 5
        }
        
        summary = detector.get_summary(result)
        
        assert "40" in summary or "40.0%" in summary
        assert "낮음" in summary or "❌" in summary
        assert "3" in summary 
        assert "⚠️" in summary  
    
    def test_empty_source_nodes(self):
        detector = create_detector()
        
        source_nodes = []
        answer = "Some answer text."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["overall_confidence"] < 0.6
        assert result["is_reliable"] is False
    
    def test_korean_text_handling(self):
        detector = create_detector()
        
        source_nodes = [
            {"content": "TreeRAG는 계층적 문서 검색 시스템입니다."}
        ]
        
        answer = "TreeRAG는 계층적 문서 검색 시스템입니다."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["overall_confidence"] >= 0.8
        assert result["is_reliable"] is True
    
    def test_multiple_source_nodes(self):
        detector = create_detector()
        
        source_nodes = [
            {"content": "First fact about TreeRAG."},
            {"content": "Second fact about indexing."},
            {"content": "Third fact about retrieval."}
        ]
        
        answer = "TreeRAG uses indexing and retrieval mechanisms."
        
        result = detector.detect(answer, source_nodes)
        
        assert result["overall_confidence"] > 0.5
    
    def test_short_sentences_ignored(self):
        detector = create_detector()
        
        source_nodes = [{"content": "Main content here."}]
        answer = "Yes. Main content here."
        
        result = detector.detect(answer, source_nodes)

        assert result["total_sentences"] == 1
    
    def test_different_thresholds(self):
        source_nodes = [{"content": "TreeRAG is a system."}]
        answer = "TreeRAG is a system for documents."
        
        strict_detector = create_detector(confidence_threshold=0.9)
        strict_result = strict_detector.detect(answer, source_nodes)
        
        lenient_detector = create_detector(confidence_threshold=0.5)
        lenient_result = lenient_detector.detect(answer, source_nodes)
        
        assert strict_result["overall_confidence"] == lenient_result["overall_confidence"]
        
        if strict_result["overall_confidence"] < 0.9:
            assert strict_result["is_reliable"] is False
        if lenient_result["overall_confidence"] >= 0.5:
            assert lenient_result["is_reliable"] is True
    
    def test_sentence_analysis_structure(self):
        detector = create_detector()
        
        source_nodes = [{"content": "Test content."}]
        answer = "Test content. More details."
        
        result = detector.detect(answer, source_nodes)
        
        assert "sentence_analysis" in result
        assert isinstance(result["sentence_analysis"], list)
        
        if result["sentence_analysis"]:
            sentence_data = result["sentence_analysis"][0]
            assert "sentence" in sentence_data
            assert "confidence" in sentence_data
            assert "is_grounded" in sentence_data
            assert isinstance(sentence_data["confidence"], float)
            assert isinstance(sentence_data["is_grounded"], bool)
