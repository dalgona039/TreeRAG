"""
Hallucination Detection Module

Detects potential hallucinations in LLM responses by comparing
generated content against source documents.
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detects potential hallucinations in LLM-generated responses."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        sentence_threshold: float | None = None,
        overall_threshold: float | None = None,
    ):
        """
        Args:
            confidence_threshold: Backward-compatible default threshold
            sentence_threshold: Threshold for sentence-level grounding
            overall_threshold: Threshold for overall answer reliability
        """
        self.sentence_threshold = sentence_threshold if sentence_threshold is not None else confidence_threshold
        self.overall_threshold = overall_threshold if overall_threshold is not None else confidence_threshold
        self.confidence_threshold = self.sentence_threshold
    
    def detect(self, answer: str, source_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze answer for potential hallucinations.
        
        Args:
            answer: Generated answer text
            source_nodes: List of source nodes with 'content' field
        
        Returns:
            Detection result with overall confidence and sentence-level analysis
        """
        source_text = self._build_source_text(source_nodes)
        
        logger.debug("Hallucination Detection:")
        logger.debug(f"  - Source nodes count: {len(source_nodes)}")
        logger.debug(f"  - Source text length: {len(source_text)} chars")
        logger.debug(f"  - Answer length: {len(answer)} chars")
        if len(source_text) > 0:
            logger.debug(f"  - Source text sample: {source_text[:200]}...")
        
        sentences = self._split_sentences(answer)
        
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            confidence = self._calculate_sentence_confidence(sentence, source_text)
            sentence_scores.append({
                "sentence": sentence,
                "confidence": confidence,
                "is_grounded": confidence >= self.sentence_threshold
            })
        
        if sentence_scores:
            overall_confidence = sum(s["confidence"] for s in sentence_scores) / len(sentence_scores)
        else:
            overall_confidence = 0.0
        
        hallucinated = [s for s in sentence_scores if not s["is_grounded"]]
        
        logger.debug(f"  - Total sentences analyzed: {len(sentence_scores)}")
        logger.debug(f"  - Overall confidence: {overall_confidence:.3f}")
        logger.debug(f"  - Hallucinated sentences: {len(hallucinated)}")
        logger.debug(f"  - Is reliable: {overall_confidence >= self.overall_threshold}")
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "is_reliable": overall_confidence >= self.overall_threshold,
            "sentence_analysis": sentence_scores,
            "hallucinated_count": len(hallucinated),
            "total_sentences": len(sentence_scores),
            "hallucinated_sentences": hallucinated,
            "thresholds": {
                "sentence": self.sentence_threshold,
                "overall": self.overall_threshold,
            }
        }

    def _build_source_text(self, source_nodes: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []

        for node in source_nodes:
            if not isinstance(node, dict):
                continue

            content = str(node.get("content", "") or "").strip()
            summary = str(node.get("summary", "") or "").strip()
            text = str(node.get("text", "") or "").strip()
            title = str(node.get("title", "") or "").strip()

            merged = " ".join(part for part in [title, content, summary, text] if part)
            if merged:
                chunks.append(merged)

        return " ".join(chunks)
    
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
        
        # Short sentences get moderate confidence
        if len(sentence_lower) < 10:
            return 0.75
        
        # Exact match
        if sentence_lower in source_lower:
            return 1.0
        
        # Use max score approach with weighted signals
        scores = []
        
        # 1. Citation presence (strong signal)
        citation_patterns = [
            r'\[.+?,\s*p\.\d+(?:[-–]\d+)?\]',
            r'\[.+?,\s*pp\.\d+(?:[-–]\d+)?\]',
            r'section\s+\d+',
            r'chapter\s+\d+',
            r'table\s+\d+'
        ]
        has_citation = any(re.search(pattern, sentence_lower) for pattern in citation_patterns)
        if has_citation:
            scores.append(0.85)
        
        # 2. Word overlap with weights
        sentence_words = re.findall(r'\w+', sentence_lower)
        source_words = re.findall(r'\w+', source_lower)
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                     'this', 'that', 'these', 'those', 'it', 'its', 'which', 'what', 'who',
                     '은', '는', '이', '가', '을', '를', '의', '에', '도', '와', '과', '으로', '로'}
        
        sentence_words_filtered = [w for w in sentence_words if w not in stop_words and len(w) > 1]
        source_words_set = set(source_words)
        
        if sentence_words_filtered:
            matched_words = 0
            total_weight = 0
            
            for word in sentence_words_filtered:
                weight = 1.0
                # Numbers and long words are important
                if re.match(r'\d+', word):
                    weight = 2.5
                elif len(word) > 8:
                    weight = 2.0
                elif len(word) > 5:
                    weight = 1.5
                
                total_weight += weight
                if word in source_words_set:
                    matched_words += weight
            
            if total_weight > 0:
                word_overlap = matched_words / total_weight
                # Word overlap is a strong signal, weight it heavily
                scores.append(word_overlap)
                scores.append(word_overlap)  # Double weight
        
        # 3. Bigram overlap (moderate signal)
        bigrams_sent = [' '.join(sentence_words[i:i+2]) for i in range(len(sentence_words)-1)]
        bigrams_src = [' '.join(source_words[i:i+2]) for i in range(len(source_words)-1)]
        if bigrams_sent:
            bigram_overlap = len([b for b in bigrams_sent if b in bigrams_src]) / len(bigrams_sent)
            if bigram_overlap > 0:
                scores.append(bigram_overlap * 1.2)  # Boost bigram signals
        
        # 4. Trigram overlap (strong signal when present)
        trigrams_sent = [' '.join(sentence_words[i:i+3]) for i in range(len(sentence_words)-2)]
        trigrams_src = [' '.join(source_words[i:i+3]) for i in range(len(source_words)-2)]
        if trigrams_sent:
            trigram_overlap = len([t for t in trigrams_sent if t in trigrams_src]) / len(trigrams_sent)
            if trigram_overlap > 0:
                scores.append(trigram_overlap * 1.5)  # Strong boost for trigram matches
        
        # 5. Character chunk overlap
        chunk_size = 15
        chunks = [sentence_lower[i:i+chunk_size] for i in range(0, len(sentence_lower), chunk_size//2)]
        chunk_matches = sum(1 for chunk in chunks if len(chunk) > 5 and chunk in source_lower)
        if chunks:
            chunk_score = chunk_matches / len(chunks)
            if chunk_score > 0:
                scores.append(chunk_score)
        
        # Calculate final confidence: use weighted average with baseline
        if scores:
            # Take the max of average and best signal (with weight)
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            # Blend: 60% max, 40% average for more balanced scoring
            confidence = 0.6 * max_score + 0.4 * avg_score
        else:
            # No matches found, but set baseline to 0.3 instead of 0
            confidence = 0.3
        
        # Ensure minimum confidence of 0.35 for any response
        confidence = max(confidence, 0.35)
        
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


def create_detector(
    confidence_threshold: float = 0.5,
    sentence_threshold: float | None = None,
    overall_threshold: float | None = None,
) -> HallucinationDetector:
    """
    Factory function to create a hallucination detector.
    
    Args:
        confidence_threshold: Backward-compatible default threshold
        sentence_threshold: Sentence-level threshold
        overall_threshold: Overall reliability threshold
    
    Returns:
        Configured HallucinationDetector instance
    """
    return HallucinationDetector(
        confidence_threshold=confidence_threshold,
        sentence_threshold=sentence_threshold,
        overall_threshold=overall_threshold,
    )
