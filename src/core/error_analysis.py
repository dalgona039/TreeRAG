import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path


class ErrorCategory(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_NOISE = "retrieval_noise"
    RETRIEVAL_PARTIAL = "retrieval_partial"
    REASONING_INCORRECT = "reasoning_incorrect"
    REASONING_INCOMPLETE = "reasoning_incomplete"
    REASONING_OVERCLAIMED = "reasoning_overclaimed"
    GENERATION_HALLUCINATION = "generation_hallucination"
    GENERATION_CONTRADICTION = "generation_contradiction"
    GENERATION_MISATTRIBUTION = "generation_misattribution"
    CONTEXT_INSUFFICIENT = "context_insufficient"
    CONTEXT_AMBIGUOUS = "context_ambiguous"
    CONTEXT_OUTDATED = "context_outdated"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    COSMETIC = "cosmetic"


@dataclass
class ErrorInstance:
    error_id: str
    category: ErrorCategory
    severity: SeverityLevel
    query_id: str
    description: str
    evidence: str
    expected: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "query_id": self.query_id,
            "description": self.description,
            "evidence": self.evidence,
            "expected": self.expected,
            "confidence": self.confidence
        }


@dataclass
class ErrorAnalysisResult:
    total_queries: int = 0
    queries_with_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    errors: List[ErrorInstance] = field(default_factory=list)
    category_severity_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "queries_with_errors": self.queries_with_errors,
            "error_rate": self.error_rate,
            "errors_by_category": self.errors_by_category,
            "errors_by_severity": self.errors_by_severity,
            "category_severity_matrix": self.category_severity_matrix,
            "total_errors": len(self.errors)
        }


@dataclass
class CalibrationBin:
    bin_start: float
    bin_end: float
    n_samples: int
    mean_confidence: float
    accuracy: float
    
    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)


@dataclass
class CalibrationResult:
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    average_confidence: float = 0.0
    overall_accuracy: float = 0.0
    brier_score: float = 0.0
    bins: List[CalibrationBin] = field(default_factory=list)
    confidence_values: List[float] = field(default_factory=list)
    accuracy_values: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.expected_calibration_error,
            "mce": self.maximum_calibration_error,
            "avg_confidence": self.average_confidence,
            "accuracy": self.overall_accuracy,
            "brier_score": self.brier_score,
            "n_bins": len(self.bins),
            "bins": [
                {
                    "range": [b.bin_start, b.bin_end],
                    "n_samples": b.n_samples,
                    "confidence": b.mean_confidence,
                    "accuracy": b.accuracy,
                    "gap": b.gap
                }
                for b in self.bins
            ]
        }


class ErrorAnalyzer:
    def __init__(self):
        self.errors: List[ErrorInstance] = []
        self._error_counter = 0
    
    def classify_error(
        self,
        query_id: str,
        answer: str,
        reference: str,
        context: str,
        confidence: float = 0.0
    ) -> List[ErrorInstance]:
        errors = []
        
        if self._detect_retrieval_miss(answer, reference, context):
            errors.append(self._create_error(
                category=ErrorCategory.RETRIEVAL_MISS,
                severity=SeverityLevel.CRITICAL,
                query_id=query_id,
                description="Relevant information not in context",
                evidence=f"Context doesn't contain: {reference[:100]}...",
                expected=reference,
                confidence=confidence
            ))
        
        hallucination = self._detect_hallucination(answer, context)
        if hallucination:
            errors.append(self._create_error(
                category=ErrorCategory.GENERATION_HALLUCINATION,
                severity=SeverityLevel.MAJOR,
                query_id=query_id,
                description="Answer contains information not in context",
                evidence=hallucination,
                confidence=confidence
            ))
        
        if self._detect_overclaiming(answer, context):
            errors.append(self._create_error(
                category=ErrorCategory.REASONING_OVERCLAIMED,
                severity=SeverityLevel.MINOR,
                query_id=query_id,
                description="Answer makes claims beyond evidence",
                evidence=answer[:200],
                confidence=confidence
            ))
        
        if self._detect_contradiction(answer, context):
            errors.append(self._create_error(
                category=ErrorCategory.GENERATION_CONTRADICTION,
                severity=SeverityLevel.CRITICAL,
                query_id=query_id,
                description="Answer contradicts context",
                evidence=answer[:200],
                confidence=confidence
            ))
        
        self.errors.extend(errors)
        return errors
    
    def _create_error(self, **kwargs) -> ErrorInstance:
        self._error_counter += 1
        return ErrorInstance(
            error_id=f"err_{self._error_counter:04d}",
            **kwargs
        )
    
    def _detect_retrieval_miss(
        self,
        answer: str,
        reference: str,
        context: str
    ) -> bool:
        ref_words = set(reference.lower().split())
        context_words = set(context.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must',
                    'shall', 'can', 'need', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'into', 'through'}
        
        ref_words = ref_words - stopwords
        
        if len(ref_words) == 0:
            return False
        
        overlap = len(ref_words & context_words) / len(ref_words)
        return overlap < 0.3
    
    def _detect_hallucination(self, answer: str, context: str) -> Optional[str]:
        import re
        
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|억|만|천|개|년|월|일)?\b'
        numbers_in_answer = re.findall(number_pattern, answer)
        
        for num in numbers_in_answer:
            if num not in context and len(num) > 1:
                return f"Number '{num}' not found in context"
        
        name_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        names_in_answer = re.findall(name_pattern, answer)
        
        for name in names_in_answer:
            if name not in context and len(name) > 3:
                return f"Name '{name}' not found in context"
        
        return None
    
    def _detect_overclaiming(self, answer: str, context: str) -> bool:
        overclaim_markers = [
            "always", "never", "definitely", "certainly", "absolutely",
            "100%", "guaranteed", "proven", "확실히", "반드시", "절대로"
        ]
        
        answer_lower = answer.lower()
        for marker in overclaim_markers:
            if marker in answer_lower and marker not in context.lower():
                return True
        
        return False
    
    def _detect_contradiction(self, answer: str, context: str) -> bool:
        negation_pairs = [
            ("is not", "is"),
            ("are not", "are"),
            ("was not", "was"),
            ("doesn't", "does"),
            ("didn't", "did"),
            ("cannot", "can"),
            ("isn't", "is"),
            ("아니", "맞"),
            ("없", "있"),
        ]
        
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        for neg, pos in negation_pairs:
            if neg in answer_lower and pos in context_lower:
                return True
        
        return False
    
    def analyze(self, queries_with_errors: int, total_queries: int) -> ErrorAnalysisResult:
        result = ErrorAnalysisResult(
            total_queries=total_queries,
            queries_with_errors=queries_with_errors,
            errors=self.errors
        )
        
        for error in self.errors:
            cat = error.category.value
            result.errors_by_category[cat] = result.errors_by_category.get(cat, 0) + 1
            
            sev = error.severity.value
            result.errors_by_severity[sev] = result.errors_by_severity.get(sev, 0) + 1
            if cat not in result.category_severity_matrix:
                result.category_severity_matrix[cat] = {}
            result.category_severity_matrix[cat][sev] = \
                result.category_severity_matrix[cat].get(sev, 0) + 1
        result.error_rate = queries_with_errors / total_queries if total_queries > 0 else 0
        
        return result


class ConfidenceCalibrator:
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def analyze(
        self,
        confidences: List[float],
        correctness: List[bool]
    ) -> CalibrationResult:
        if len(confidences) != len(correctness):
            raise ValueError("Confidence and correctness lists must have same length")
        
        if not confidences:
            return CalibrationResult()
        
        result = CalibrationResult()
        result.average_confidence = sum(confidences) / len(confidences)
        result.overall_accuracy = sum(correctness) / len(correctness)
        
        result.brier_score = sum(
            (c - int(correct)) ** 2
            for c, correct in zip(confidences, correctness)
        ) / len(confidences)
        bin_boundaries = [i / self.n_bins for i in range(self.n_bins + 1)]
        bins = []
        
        for i in range(self.n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]
            
            bin_samples = [
                (c, correct)
                for c, correct in zip(confidences, correctness)
                if bin_start <= c < bin_end or (i == self.n_bins - 1 and c == bin_end)
            ]
            
            if bin_samples:
                bin_confidences = [c for c, _ in bin_samples]
                bin_correct = [correct for _, correct in bin_samples]
                
                bins.append(CalibrationBin(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    n_samples=len(bin_samples),
                    mean_confidence=sum(bin_confidences) / len(bin_confidences),
                    accuracy=sum(bin_correct) / len(bin_correct)
                ))
        
        result.bins = bins
        
        total_samples = len(confidences)
        ece = sum(
            (bin.n_samples / total_samples) * bin.gap
            for bin in bins
        )
        result.expected_calibration_error = ece
        if bins:
            result.maximum_calibration_error = max(bin.gap for bin in bins)
        result.confidence_values = [bin.mean_confidence for bin in bins]
        result.accuracy_values = [bin.accuracy for bin in bins]
        
        return result
    
    def calibrate_temperature(
        self,
        confidences: List[float],
        correctness: List[bool],
        n_iterations: int = 100
    ) -> Tuple[float, List[float]]:
        def apply_temperature(conf: float, temp: float) -> float:
            # Convert to logit
            conf = max(min(conf, 0.999), 0.001)
            logit = math.log(conf / (1 - conf))
            scaled_logit = logit / temp
            return 1 / (1 + math.exp(-scaled_logit))
        
        def compute_nll(temp: float) -> float:
            nll = 0.0
            for conf, correct in zip(confidences, correctness):
                cal_conf = apply_temperature(conf, temp)
                if correct:
                    nll -= math.log(max(cal_conf, 1e-10))
                else:
                    nll -= math.log(max(1 - cal_conf, 1e-10))
            return nll / len(confidences)
        
        best_temp = 1.0
        best_nll = float('inf')
        for temp in [0.1 * i for i in range(1, 51)]:
            nll = compute_nll(temp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        calibrated = [apply_temperature(c, best_temp) for c in confidences]
        
        return best_temp, calibrated
    
    def reliability_diagram_data(
        self,
        result: CalibrationResult
    ) -> Dict[str, Any]:
        return {
            "bins": [
                {
                    "confidence": bin.mean_confidence,
                    "accuracy": bin.accuracy,
                    "n_samples": bin.n_samples,
                    "gap": bin.gap
                }
                for bin in result.bins
            ],
            "perfect_calibration": list(range(11)),
            "ece": result.expected_calibration_error,
            "mce": result.maximum_calibration_error
        }


class HallucinationQuantifier:
    
    def __init__(self):
        self.hallucinations: List[Dict[str, Any]] = []
    
    def detect(
        self,
        query_id: str,
        answer: str,
        context: str,
        confidence: float
    ) -> Dict[str, Any]:
        result = {
            "query_id": query_id,
            "confidence": confidence,
            "hallucinations": [],
            "hallucination_score": 0.0
        }
        
        claims = self._extract_claims(answer)
        
        unsupported_claims = 0
        total_claims = len(claims)
        
        for claim in claims:
            is_supported, support_type = self._verify_claim(claim, context)
            
            if not is_supported:
                unsupported_claims += 1
                result["hallucinations"].append({
                    "claim": claim,
                    "type": "unsupported",
                    "severity": self._assess_severity(claim)
                })
        
        if total_claims > 0:
            result["hallucination_score"] = unsupported_claims / total_claims
        
        result["total_claims"] = total_claims
        result["unsupported_claims"] = unsupported_claims
        
        self.hallucinations.append(result)
        return result
    
    def _extract_claims(self, text: str) -> List[str]:
        import re
        
        sentences = re.split(r'[.!?]\s+|[。！？]', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                claims.append(sentence)
        
        return claims
    
    def _verify_claim(self, claim: str, context: str) -> Tuple[bool, str]:
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                    'have', 'has', 'had', 'do', 'does', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'this',
                    'that', 'these', 'those', 'it', 'its'}
        
        claim_words = claim_words - stopwords
        
        if not claim_words:
            return True, "empty"
        
        overlap = len(claim_words & context_words) / len(claim_words)
        
        if overlap >= 0.5:
            return True, "full_support"
        elif overlap >= 0.3:
            return True, "partial_support"
        else:
            return False, "unsupported"
    
    def _assess_severity(self, claim: str) -> str:
        import re
        
        if re.search(r'\d+', claim):
            return "high"
        if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', claim):
            return "high"
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'must']
        if any(word in claim.lower() for word in absolute_words):
            return "medium"
        
        return "low"
    
    def compute_metrics(self) -> Dict[str, Any]:
        if not self.hallucinations:
            return {}
        
        total_queries = len(self.hallucinations)
        total_claims = sum(h["total_claims"] for h in self.hallucinations)
        total_unsupported = sum(h["unsupported_claims"] for h in self.hallucinations)
        
        queries_with_hallucination = sum(
            1 for h in self.hallucinations if h["hallucination_score"] > 0
        )
        
        confidences = [h["confidence"] for h in self.hallucinations]
        hal_scores = [h["hallucination_score"] for h in self.hallucinations]
        correlation = self._compute_correlation(confidences, hal_scores)
        
        return {
            "total_queries": total_queries,
            "total_claims": total_claims,
            "total_unsupported": total_unsupported,
            "hallucination_rate": total_unsupported / total_claims if total_claims > 0 else 0,
            "queries_with_hallucination": queries_with_hallucination,
            "query_hallucination_rate": queries_with_hallucination / total_queries,
            "avg_hallucination_score": sum(hal_scores) / len(hal_scores),
            "confidence_hallucination_correlation": correlation
        }
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(var_x * var_y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def generate_error_report(
    error_result: ErrorAnalysisResult,
    calibration_result: CalibrationResult,
    hallucination_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "summary": {
            "total_queries": error_result.total_queries,
            "error_rate": error_result.error_rate,
            "calibration_error": calibration_result.expected_calibration_error,
            "hallucination_rate": hallucination_metrics.get("hallucination_rate", 0)
        },
        "error_analysis": error_result.to_dict(),
        "calibration": calibration_result.to_dict(),
        "hallucination": hallucination_metrics,
        "recommendations": _generate_recommendations(
            error_result, calibration_result, hallucination_metrics
        )
    }


def _generate_recommendations(
    error_result: ErrorAnalysisResult,
    calibration_result: CalibrationResult,
    hallucination_metrics: Dict[str, Any]
) -> List[str]:
    recommendations = []
    
    if error_result.errors_by_category.get(ErrorCategory.RETRIEVAL_MISS.value, 0) > 0:
        recommendations.append(
            "Consider expanding retrieval scope or using query expansion"
        )
    
    if error_result.errors_by_category.get(ErrorCategory.GENERATION_HALLUCINATION.value, 0) > 0:
        recommendations.append(
            "Implement stronger grounding constraints in generation"
        )
    
    if calibration_result.expected_calibration_error > 0.1:
        recommendations.append(
            f"Apply temperature scaling (ECE={calibration_result.expected_calibration_error:.3f})"
        )
    
    if calibration_result.average_confidence > calibration_result.overall_accuracy + 0.1:
        recommendations.append(
            "Model is overconfident - consider recalibration"
        )
    
    hal_rate = hallucination_metrics.get("hallucination_rate", 0)
    if hal_rate > 0.2:
        recommendations.append(
            f"High hallucination rate ({hal_rate:.1%}) - strengthen context verification"
        )
    
    correlation = hallucination_metrics.get("confidence_hallucination_correlation", 0)
    if correlation < -0.3:
        recommendations.append(
            "Confidence inversely correlates with hallucination - use for filtering"
        )
    
    return recommendations
