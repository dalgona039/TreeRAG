"""
Fidelity Metrics for TreeRAG Benchmarking.

Measures answer quality and faithfulness:
- Hallucination detection and quantification
- Answer groundedness (claims supported by context)
- Factual consistency
- Citation accuracy
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum


class HallucinationType(str, Enum):
    """Types of hallucinations."""
    UNSUPPORTED_CLAIM = "unsupported_claim"  # Claim not in context
    CONTRADICTORY = "contradictory"  # Contradicts context
    FABRICATED_ENTITY = "fabricated_entity"  # Made-up names/numbers
    EXTRAPOLATION = "extrapolation"  # Goes beyond context
    TEMPORAL_ERROR = "temporal_error"  # Wrong dates/sequence
    QUANTITATIVE_ERROR = "quantitative_error"  # Wrong numbers


@dataclass
class Claim:
    """Single factual claim in an answer."""
    text: str
    claim_type: str  # fact, number, entity, temporal
    grounded: bool  # Supported by context
    source_span: Optional[str] = None  # Supporting text from context
    confidence: float = 0.0
    hallucination_type: Optional[HallucinationType] = None


@dataclass
class FidelityAnalysis:
    """Analysis of a single answer's fidelity."""
    query_id: str
    answer_text: str
    context_text: str
    claims: List[Claim] = field(default_factory=list)
    
    @property
    def total_claims(self) -> int:
        """Total number of claims."""
        return len(self.claims)
    
    @property
    def grounded_claims(self) -> int:
        """Number of grounded claims."""
        return sum(1 for c in self.claims if c.grounded)
    
    @property
    def hallucinated_claims(self) -> int:
        """Number of hallucinated claims."""
        return sum(1 for c in self.claims if not c.grounded)
    
    @property
    def groundedness_score(self) -> float:
        """Fraction of claims that are grounded."""
        if not self.claims:
            return 1.0
        return self.grounded_claims / self.total_claims
    
    @property
    def hallucination_rate(self) -> float:
        """Fraction of claims that are hallucinated."""
        return 1.0 - self.groundedness_score


@dataclass
class FidelityResult:
    """Aggregated fidelity metrics."""
    # Overall scores
    groundedness_mean: float = 0.0
    groundedness_std: float = 0.0
    hallucination_rate_mean: float = 0.0
    
    # By claim type
    claims_by_type: Dict[str, int] = field(default_factory=dict)
    groundedness_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Hallucination breakdown
    hallucinations_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Per-query details
    per_query_groundedness: Dict[str, float] = field(default_factory=dict)
    per_query_claims: Dict[str, int] = field(default_factory=dict)
    
    # Statistics
    total_claims: int = 0
    total_grounded: int = 0
    total_hallucinated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "groundedness": {
                "mean": self.groundedness_mean,
                "std": self.groundedness_std,
                "by_type": self.groundedness_by_type
            },
            "hallucination_rate": self.hallucination_rate_mean,
            "hallucinations_by_type": self.hallucinations_by_type,
            "claims": {
                "total": self.total_claims,
                "grounded": self.total_grounded,
                "hallucinated": self.total_hallucinated,
                "by_type": self.claims_by_type
            }
        }


class FidelityMetrics:
    """
    Calculator for answer fidelity metrics.
    
    Measures how well answers are grounded in the retrieved context
    and detects various types of hallucinations.
    """
    
    def __init__(self, use_llm: bool = False, llm_client: Optional[Any] = None):
        """
        Initialize fidelity metrics calculator.
        
        Args:
            use_llm: Whether to use LLM for claim extraction/verification
            llm_client: LLM client for advanced analysis
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        
        # Patterns for entity/number extraction
        self._number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:%|억|만|천|백|개|년|월|일)?\b')
        self._entity_pattern = re.compile(r'(?:^|[^가-힣])[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*')
        self._date_pattern = re.compile(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{4}년\s*\d{1,2}월|\d{1,2}월\s*\d{1,2}일')
    
    def extract_claims_simple(self, answer: str) -> List[Claim]:
        """
        Extract claims from answer using simple heuristics.
        
        Args:
            answer: Answer text to analyze
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+|[。！？]', answer)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Determine claim type
            claim_type = "fact"
            
            if self._number_pattern.search(sentence):
                claim_type = "number"
            elif self._date_pattern.search(sentence):
                claim_type = "temporal"
            elif self._entity_pattern.search(sentence):
                claim_type = "entity"
            
            claims.append(Claim(
                text=sentence,
                claim_type=claim_type,
                grounded=False  # Will be verified later
            ))
        
        return claims
    
    def verify_claim_simple(self, claim: Claim, context: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if a claim is supported by context using simple matching.
        
        Args:
            claim: Claim to verify
            context: Context to check against
            
        Returns:
            Tuple of (is_grounded, supporting_span)
        """
        claim_text = claim.text.lower()
        context_lower = context.lower()
        
        # Extract key terms from claim
        words = set(re.findall(r'\b\w{3,}\b', claim_text))
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과'}
        words = words - stopwords
        
        if not words:
            return True, None  # No meaningful words to check
        
        # Check overlap
        context_words = set(re.findall(r'\b\w{3,}\b', context_lower))
        overlap = words & context_words
        overlap_ratio = len(overlap) / len(words) if words else 0
        
        # For numbers, check exact match
        if claim.claim_type == "number":
            numbers_in_claim = self._number_pattern.findall(claim.text)
            for num in numbers_in_claim:
                if num in context:
                    return True, f"Found: {num}"
            # Low overlap and no matching numbers = likely hallucination
            if overlap_ratio < 0.5:
                return False, None
        
        # For entities, check if entity name appears
        if claim.claim_type == "entity":
            entities = self._entity_pattern.findall(claim.text)
            for entity in entities:
                if entity.lower() in context_lower:
                    return True, f"Found entity: {entity}"
        
        # General: require significant overlap
        is_grounded = overlap_ratio >= 0.4
        
        return is_grounded, f"Overlap: {overlap_ratio:.2f}" if is_grounded else None
    
    def analyze_answer(
        self,
        query_id: str,
        answer: str,
        context: str
    ) -> FidelityAnalysis:
        """
        Analyze fidelity of an answer.
        
        Args:
            query_id: Query identifier
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Fidelity analysis
        """
        # Extract claims
        claims = self.extract_claims_simple(answer)
        
        # Verify each claim
        for claim in claims:
            is_grounded, source_span = self.verify_claim_simple(claim, context)
            claim.grounded = is_grounded
            claim.source_span = source_span
            
            if not is_grounded:
                # Classify hallucination type
                claim.hallucination_type = self._classify_hallucination(claim, context)
        
        return FidelityAnalysis(
            query_id=query_id,
            answer_text=answer,
            context_text=context,
            claims=claims
        )
    
    def _classify_hallucination(self, claim: Claim, context: str) -> HallucinationType:
        """Classify the type of hallucination."""
        claim_text = claim.text.lower()
        context_lower = context.lower()
        
        # Check for contradiction
        negation_pairs = [
            ('not', 'is'), ('no', 'yes'), ('never', 'always'),
            ('없', '있'), ('아니', '맞')
        ]
        for neg, pos in negation_pairs:
            if neg in claim_text and pos in context_lower:
                return HallucinationType.CONTRADICTORY
        
        # Check for fabricated numbers
        if claim.claim_type == "number":
            return HallucinationType.QUANTITATIVE_ERROR
        
        # Check for temporal errors
        if claim.claim_type == "temporal":
            return HallucinationType.TEMPORAL_ERROR
        
        # Check for fabricated entities
        if claim.claim_type == "entity":
            return HallucinationType.FABRICATED_ENTITY
        
        # Default: unsupported claim
        return HallucinationType.UNSUPPORTED_CLAIM
    
    def compute_metrics(
        self,
        analyses: List[FidelityAnalysis]
    ) -> FidelityResult:
        """
        Compute aggregated fidelity metrics.
        
        Args:
            analyses: List of fidelity analyses
            
        Returns:
            Aggregated fidelity result
        """
        import statistics
        
        result = FidelityResult()
        
        if not analyses:
            return result
        
        # Aggregate scores
        groundedness_scores = [a.groundedness_score for a in analyses]
        
        result.groundedness_mean = statistics.mean(groundedness_scores)
        result.groundedness_std = statistics.stdev(groundedness_scores) if len(groundedness_scores) > 1 else 0.0
        result.hallucination_rate_mean = 1.0 - result.groundedness_mean
        
        # Count claims by type
        claims_by_type: Dict[str, int] = {}
        grounded_by_type: Dict[str, int] = {}
        hallucinations_by_type: Dict[str, int] = {}
        
        for analysis in analyses:
            result.per_query_groundedness[analysis.query_id] = analysis.groundedness_score
            result.per_query_claims[analysis.query_id] = analysis.total_claims
            
            for claim in analysis.claims:
                claims_by_type[claim.claim_type] = claims_by_type.get(claim.claim_type, 0) + 1
                
                if claim.grounded:
                    grounded_by_type[claim.claim_type] = grounded_by_type.get(claim.claim_type, 0) + 1
                else:
                    h_type = claim.hallucination_type.value if claim.hallucination_type else "unknown"
                    hallucinations_by_type[h_type] = hallucinations_by_type.get(h_type, 0) + 1
                
                result.total_claims += 1
                if claim.grounded:
                    result.total_grounded += 1
                else:
                    result.total_hallucinated += 1
        
        result.claims_by_type = claims_by_type
        result.hallucinations_by_type = hallucinations_by_type
        
        # Compute groundedness by type
        for claim_type, count in claims_by_type.items():
            grounded = grounded_by_type.get(claim_type, 0)
            result.groundedness_by_type[claim_type] = grounded / count if count > 0 else 0.0
        
        return result


class CitationAccuracy:
    """
    Measures citation accuracy in answers.
    
    Checks if citations in the answer correctly reference
    the supporting content.
    """
    
    def __init__(self):
        """Initialize citation accuracy checker."""
        # Pattern for citations like [1], [2], (Source: X)
        self._citation_pattern = re.compile(r'\[(\d+)\]|\((?:Source|출처|참고):\s*([^)]+)\)')
    
    def extract_citations(self, answer: str) -> List[Tuple[str, int]]:
        """
        Extract citations from answer.
        
        Args:
            answer: Answer text
            
        Returns:
            List of (citation_text, position) tuples
        """
        citations = []
        
        for match in self._citation_pattern.finditer(answer):
            citation = match.group(1) or match.group(2)
            citations.append((citation, match.start()))
        
        return citations
    
    def verify_citations(
        self,
        answer: str,
        sources: Dict[str, str]  # citation -> source text
    ) -> Dict[str, Any]:
        """
        Verify citation accuracy.
        
        Args:
            answer: Answer with citations
            sources: Mapping of citation to source text
            
        Returns:
            Citation accuracy analysis
        """
        citations = self.extract_citations(answer)
        
        if not citations:
            return {
                "total_citations": 0,
                "verified": 0,
                "accuracy": 1.0,  # No citations = no errors
                "details": []
            }
        
        verified = 0
        details = []
        
        for citation, pos in citations:
            # Get surrounding context in answer
            start = max(0, pos - 100)
            end = min(len(answer), pos + 100)
            surrounding = answer[start:end]
            
            # Check if citation source supports surrounding text
            source = sources.get(citation, "")
            if source:
                # Simple overlap check
                words = set(re.findall(r'\b\w{4,}\b', surrounding.lower()))
                source_words = set(re.findall(r'\b\w{4,}\b', source.lower()))
                overlap = len(words & source_words) / len(words) if words else 0
                
                is_valid = overlap > 0.3
                if is_valid:
                    verified += 1
                
                details.append({
                    "citation": citation,
                    "valid": is_valid,
                    "overlap": overlap
                })
            else:
                details.append({
                    "citation": citation,
                    "valid": False,
                    "reason": "Source not found"
                })
        
        return {
            "total_citations": len(citations),
            "verified": verified,
            "accuracy": verified / len(citations),
            "details": details
        }


def compare_fidelity(
    treerag_analyses: List[FidelityAnalysis],
    baseline_analyses: List[FidelityAnalysis]
) -> Dict[str, Any]:
    """
    Compare fidelity between TreeRAG and baseline.
    
    Args:
        treerag_analyses: Fidelity analyses for TreeRAG
        baseline_analyses: Fidelity analyses for baseline
        
    Returns:
        Comparison results
    """
    metrics = FidelityMetrics()
    
    treerag_result = metrics.compute_metrics(treerag_analyses)
    baseline_result = metrics.compute_metrics(baseline_analyses)
    
    return {
        "treerag": treerag_result.to_dict(),
        "baseline": baseline_result.to_dict(),
        "comparison": {
            "groundedness_improvement": treerag_result.groundedness_mean - baseline_result.groundedness_mean,
            "hallucination_reduction": baseline_result.hallucination_rate_mean - treerag_result.hallucination_rate_mean,
            "treerag_better": treerag_result.groundedness_mean > baseline_result.groundedness_mean
        }
    }
