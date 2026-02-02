"""
Hallucination Detection Module

Detects potential hallucinations in LLM responses by comparing
generated content against source documents.
"""
import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher


class HallucinationDetector:
    """Detects potential hallucinations in LLM-generated responses."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: Minimum confidence score (0-1) to consider valid
        """
        self.confidence_threshold = confidence_threshold
    
    def detect(self, answer: str, source_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze answer for potential hallucinations.
        
        Args:
            answer: Generated answer text
            source_nodes: List of source nodes with 'content' field
        
        Returns:
            Detection result with overall confidence and sentence-level analysis
        """
        source_text = " ".join([node.get("content", "") for node in source_nodes])
        
        sentences = self._split_sentences(answer)
        
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            confidence = self._calculate_sentence_confidence(sentence, source_text)
            sentence_scores.append({
                "sentence": sentence,
                "confidence": confidence,
                "is_grounded": confidence >= self.confidence_threshold
            })
        
        if sentence_scores:
            overall_confidence = sum(s["confidence"] for s in sentence_scores) / len(sentence_scores)
        else:
            overall_confidence = 0.0
        
        hallucinated = [s for s in sentence_scores if not s["is_grounded"]]
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "is_reliable": overall_confidence >= self.confidence_threshold,
            "sentence_analysis": sentence_scores,
            "hallucinated_count": len(hallucinated),
            "total_sentences": len(sentence_scores),
            "hallucinated_sentences": hallucinated
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentence_confidence(self, sentence: str, source_text: str) -> float:
        """
        Calculate confidence that a sentence is grounded in source.
        
        Uses multiple signals:
        1. Direct substring matching
        2. Semantic similarity (word overlap)
        3. N-gram overlap
        """
        sentence_lower = sentence.lower()
        source_lower = source_text.lower()
        
        if sentence_lower in source_lower:
            return 1.0
        
        sentence_words = set(re.findall(r'\w+', sentence_lower))
        source_words = set(re.findall(r'\w+', source_lower))
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had'}
        sentence_words = sentence_words - stop_words
        source_words = source_words - stop_words
        
        if not sentence_words:
            return 0.5
        
        word_overlap = len(sentence_words & source_words) / len(sentence_words)
        
        similarity = SequenceMatcher(None, sentence_lower, source_lower).ratio()
        
        confidence = (word_overlap * 0.7) + (similarity * 0.3)
        
        return min(confidence, 1.0)
    
    def format_with_warnings(self, answer: str, detection_result: Dict[str, Any]) -> str:
        """
        Format answer with warning markers for low-confidence sections.
        
        Args:
            answer: Original answer
            detection_result: Result from detect()
        
        Returns:
            Formatted answer with ⚠️ warnings
        """
        if detection_result["is_reliable"]:
            return answer
        
        formatted = answer
        for hal in detection_result["hallucinated_sentences"]:
            sentence = hal["sentence"]
            confidence = hal["confidence"]
            warning = f" ⚠️ (신뢰도: {confidence:.0%})"
            formatted = formatted.replace(sentence, sentence + warning)
        
        return formatted
    
    def get_summary(self, detection_result: Dict[str, Any]) -> str:
        """
        Get human-readable summary of detection result.
        
        Args:
            detection_result: Result from detect()
        
        Returns:
            Summary string
        """
        confidence = detection_result["overall_confidence"]
        hallucinated = detection_result["hallucinated_count"]
        total = detection_result["total_sentences"]
        
        if confidence >= 0.9:
            level = "매우 높음 ✅"
        elif confidence >= 0.7:
            level = "높음 ✓"
        elif confidence >= 0.5:
            level = "중간 ⚠️"
        else:
            level = "낮음 ❌"
        
        summary = f"""
신뢰도 분석:
- 전체 신뢰도: {confidence:.1%} ({level})
- 분석된 문장: {total}개
- 근거 불충분: {hallucinated}개
"""
        
        if hallucinated > 0:
            summary += f"\n⚠️ {hallucinated}개 문장이 원문에서 충분한 근거를 찾을 수 없습니다."
        
        return summary.strip()


def create_detector(confidence_threshold: float = 0.6) -> HallucinationDetector:
    """
    Factory function to create a hallucination detector.
    
    Args:
        confidence_threshold: Minimum confidence (0-1) to consider valid
    
    Returns:
        Configured HallucinationDetector instance
    """
    return HallucinationDetector(confidence_threshold=confidence_threshold)
