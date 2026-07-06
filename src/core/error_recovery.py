import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class FilteringDecision:
    is_relevant: bool
    confidence: float
    reason: str
    llm_score: Optional[float] = None
    keyword_score: Optional[float] = None
    combined_score: Optional[float] = None


class ErrorRecoveryFilter:
    
    def __init__(
        self,
        llm_weight: float = 0.7,
        keyword_weight: float = 0.3,
        confidence_threshold: float = 0.6,
        recovery_threshold: float = 0.5
    ):
        self.llm_weight = llm_weight
        self.keyword_weight = keyword_weight
        self.confidence_threshold = confidence_threshold
        self.recovery_threshold = recovery_threshold
        self.filtering_history: List[Dict[str, Any]] = []
        self.false_negatives_detected = 0
    
    def dual_stage_filter(
        self,
        node: Dict[str, Any],
        query: str,
        parent_context: str,
        depth: int,
        llm_check_fn = None,
        threshold: float = 0.5
    ) -> FilteringDecision:
        
        if depth == 0:
            return FilteringDecision(
                is_relevant=True,
                confidence=1.0,
                reason="Root node always relevant"
            )
        
        title = node.get("title", "")
        summary = node.get("summary", "")
        
        if len(summary) < 20 and len(title) < 10:
            decision = FilteringDecision(
                is_relevant=False,
                confidence=0.95,
                reason="Content too sparse to evaluate"
            )
            self.filtering_history.append({
                'node_id': node.get('id', 'unknown'),
                'title': title,
                'llm_score': 0.0,
                'keyword_score': 0.0,
                'combined_score': 0.0,
                'confidence': 0.95,
                'is_relevant': False,
                'depth': depth
            })
            return decision
        
        llm_score, llm_confidence = self._llm_evaluate(
            node, query, parent_context, llm_check_fn
        ) if llm_check_fn else (0.5, 0.0)
        
        keyword_score = self._keyword_evaluate(title, summary, query)
        keyword_confidence = 1.0 if keyword_score > 0.5 else 0.3
        
        combined_score = (
            self.llm_weight * llm_score +
            self.keyword_weight * keyword_score
        )
        
        confidence = (
            self.llm_weight * llm_confidence +
            self.keyword_weight * keyword_confidence
        )
        
        is_relevant = combined_score > threshold
        
        self.filtering_history.append({
            'node_id': node.get('id', 'unknown'),
            'title': title,
            'llm_score': llm_score,
            'keyword_score': keyword_score,
            'combined_score': combined_score,
            'confidence': confidence,
            'is_relevant': is_relevant,
            'depth': depth
        })
        
        decision = FilteringDecision(
            is_relevant=is_relevant,
            confidence=confidence,
            reason=self._generate_reason(llm_score, keyword_score, combined_score),
            llm_score=llm_score,
            keyword_score=keyword_score,
            combined_score=combined_score
        )
        
        return decision
    
    def _llm_evaluate(
        self,
        node: Dict[str, Any],
        query: str,
        parent_context: str,
        llm_check_fn
    ) -> Tuple[float, float]:
        
        try:
            result = llm_check_fn(node, query, parent_context)
            
            is_relevant = result.get("relevant", False)
            confidence = result.get("confidence", 0.5)
            
            llm_score = 0.9 if is_relevant else 0.1
            
            return llm_score, float(confidence)
            
        except Exception as e:
            return 0.5, 0.3
    
    def _keyword_evaluate(self, title: str, summary: str, query: str) -> float:
        
        query_lower = query.lower()
        title_lower = title.lower()
        summary_lower = summary.lower()
        
        query_keywords = re.findall(r'\w+', query_lower)
        query_keywords = [kw for kw in query_keywords if len(kw) > 2]
        
        if not query_keywords:
            return 0.3
        
        title_matches = sum(1 for kw in query_keywords if kw in title_lower)
        summary_matches = sum(1 for kw in query_keywords if kw in summary_lower)
        
        title_score = min(title_matches / len(query_keywords), 1.0) if query_keywords else 0
        summary_score = min(summary_matches / len(query_keywords), 1.0) if query_keywords else 0
        
        combined = (title_score * 0.4 + summary_score * 0.6)
        
        return combined
    
    def _generate_reason(
        self,
        llm_score: float,
        keyword_score: float,
        combined_score: float
    ) -> str:
        
        if combined_score > 0.7:
            if llm_score > keyword_score:
                return "Strong LLM match"
            else:
                return "Strong keyword match"
        elif combined_score > 0.5:
            return "Moderate match from multiple signals"
        else:
            if llm_score > 0.5:
                return "LLM suggests relevance but weak keyword signals"
            else:
                return "Minimal relevance signals"
    
    def detect_over_filtering(
        self,
        selected_nodes: List[Dict[str, Any]],
        filtered_nodes: List[Dict[str, Any]],
        query: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        
        if len(selected_nodes) == 0 and len(filtered_nodes) > 0:
            return True, self._recover_critical_nodes(filtered_nodes, query)
        
        if len(selected_nodes) < 2 and len(filtered_nodes) > len(selected_nodes) * 2:
            return True, self._recover_critical_nodes(filtered_nodes, query)
        
        return False, []
    
    def _recover_critical_nodes(
        self,
        filtered_nodes: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        
        recovered = []
        query_lower = query.lower()
        
        for node in filtered_nodes:
            title = node.get('title', '').lower()
            summary = node.get('summary', '').lower()
            
            critical_keywords = re.findall(r'\b\w{4,}\b', query_lower)
            
            matches = sum(1 for kw in critical_keywords if kw in title)
            
            if matches >= 2 or (matches >= 1 and len(title) > 20):
                recovered.append(node)
                self.false_negatives_detected += 1
        
        return recovered[:3]
    
    def adaptive_threshold_adjustment(
        self,
        num_selected: int,
        num_total: int,
        query_length: int,
        depth: int = 1
    ) -> float:
        """Compute tau(q, depth): a threshold adjusted by how aggressively the
        traversal has been filtering so far (filter_rate), how specific the
        query is (query_length), and how deep in the tree we are (depth).

        Shallow nodes (depth<=1) get a lower bar to favor recall — pruning
        early costs more than a few extra LLM checks. Deep nodes (depth>=4)
        get a higher bar to favor precision, since by then the traversal has
        already committed a lot of budget and false positives are expensive
        to unwind.
        """
        base_threshold = self.confidence_threshold

        filter_rate = 1.0 - (num_selected / num_total) if num_total > 0 else 0.0

        if filter_rate > 0.9:
            rate_adj = -0.15
        elif filter_rate > 0.7:
            rate_adj = -0.1
        elif filter_rate < 0.3:
            rate_adj = 0.1
        else:
            rate_adj = 0.0

        if query_length < 10:
            length_adj = 0.1
        elif query_length > 100:
            length_adj = -0.05
        else:
            length_adj = 0.0

        if depth <= 1:
            depth_adj = -0.05
        elif depth >= 4:
            depth_adj = 0.05
        else:
            depth_adj = 0.0

        return base_threshold + rate_adj + length_adj + depth_adj
    
    def explain_filtering_decisions(self, limit: int = 10) -> str:
        
        report = "=== Filtering Decision Report ===\n\n"
        
        confident_rejects = [
            h for h in self.filtering_history
            if not h['is_relevant'] and h['confidence'] > 0.8
        ]
        
        uncertain_rejects = [
            h for h in self.filtering_history
            if not h['is_relevant'] and h['confidence'] <= 0.8
        ]
        
        report += f"Confident Rejections (confidence > 0.8): {len(confident_rejects)}\n"
        for h in confident_rejects[:limit]:
            report += f"  - {h['title']}: {h['combined_score']:.2f} confidence ({h['confidence']:.2f})\n"
        
        report += f"\nUncertain Rejections (confidence <= 0.8): {len(uncertain_rejects)}\n"
        for h in uncertain_rejects[:limit]:
            report += f"  - {h['title']}: {h['combined_score']:.2f} confidence ({h['confidence']:.2f})\n"
        
        report += f"\nFalse Negatives Detected: {self.false_negatives_detected}\n"
        
        return report
