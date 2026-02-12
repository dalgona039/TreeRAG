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
        
        Enhanced semantic similarity using:
        1. Direct substring matching
        2. Weighted word overlap (domain terms, numbers)
        3. N-gram overlap (bigrams, trigrams)
        4. Character-level similarity
        5. Citation presence boost
        """
        sentence_lower = sentence.lower()
        source_lower = source_text.lower()
        
        if len(sentence_lower) < 10:
            return 0.7
        
        if sentence_lower in source_lower:
            return 1.0
        
        signals = []
        
        citation_patterns = [r'\[.+?,\s*p\.\d+\]', r'section\s+\d+', r'chapter\s+\d+', r'table\s+\d+']
        has_citation = any(re.search(pattern, sentence_lower) for pattern in citation_patterns)
        if has_citation:
            signals.append(0.8)
        
        sentence_words = re.findall(r'\w+', sentence_lower)
        source_words = re.findall(r'\w+', source_lower)
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                     'this', 'that', 'these', 'those', 'it', 'its', 'which', 'what', 'who'}
        
        sentence_words_filtered = [w for w in sentence_words if w not in stop_words]
        source_words_set = set(source_words)
        
        if sentence_words_filtered:
            word_weights = []
            for word in sentence_words_filtered:
                weight = 1.0
                if re.match(r'\d+', word):
                    weight = 2.0
                if len(word) > 8:
                    weight = 1.5
                if word in source_words_set:
                    word_weights.append(weight)
            
            if len(sentence_words_filtered) > 0:
                weighted_overlap = sum(word_weights) / len(sentence_words_filtered)
                signals.append(weighted_overlap)
                signals.append(weighted_overlap)
        
        bigrams_sent = [' '.join(sentence_words[i:i+2]) for i in range(len(sentence_words)-1)]
        bigrams_src = [' '.join(source_words[i:i+2]) for i in range(len(source_words)-1)]
        if bigrams_sent:
            bigram_overlap = len([b for b in bigrams_sent if b in bigrams_src]) / len(bigrams_sent)
            signals.append(bigram_overlap)
        
        trigrams_sent = [' '.join(sentence_words[i:i+3]) for i in range(len(sentence_words)-2)]
        trigrams_src = [' '.join(source_words[i:i+3]) for i in range(len(source_words)-2)]
        if trigrams_sent:
            trigram_overlap = len([t for t in trigrams_sent if t in trigrams_src]) / len(trigrams_sent)
            signals.append(trigram_overlap)
        
        chunks = [sentence_lower[i:i+20] for i in range(0, len(sentence_lower), 10)]
        chunk_matches = sum(1 for chunk in chunks if chunk in source_lower)
        if chunks:
            chunk_score = chunk_matches / len(chunks)
            signals.append(chunk_score)
        
        if signals:
            confidence = sum(signals) / len(signals)
        else:
            confidence = 0.0
        
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
