
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.config import Config
from src.repositories import IndexRepository
from src.core.reasoner import TreeRAGReasoner
from src.core.contextual_compressor import ContextualCompressor
from src.utils.hallucination_detector import HallucinationDetector
from .document_router_service import DocumentRouterService


@dataclass
class NodeContext:
    id: str
    title: str
    page_ref: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class ComparisonResult:
    has_comparison: bool
    documents_compared: List[str]
    commonalities: Optional[str] = None
    differences: Optional[str] = None


@dataclass
class TraversalInfo:
    used_deep_traversal: bool
    nodes_visited: List[str]
    nodes_selected: List[Dict[str, Any]]
    max_depth: int
    max_branches: int


@dataclass
class ResolvedReference:
    title: str
    page_ref: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class HallucinationWarning:
    message: str
    overall_confidence: float
    threshold: float


@dataclass
class ChatResult:
    success: bool
    answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    comparison: Optional[ComparisonResult] = None
    traversal_info: Optional[TraversalInfo] = None
    resolved_references: Optional[List[ResolvedReference]] = None
    hallucination_warning: Optional[HallucinationWarning] = None
    error_message: Optional[str] = None


class ChatService:
    def __init__(
        self,
        index_repository: Optional[IndexRepository] = None,
        document_router: Optional[DocumentRouterService] = None,
        hallucination_threshold: float = 0.3,
        enable_compression: bool = True
    ):
        self.index_repo = index_repository or IndexRepository()
        self.document_router = document_router or DocumentRouterService(self.index_repo)
        self.hallucination_threshold = hallucination_threshold
        self.enable_compression = enable_compression
        self.compressor = ContextualCompressor() if enable_compression else None
    
    def chat(
        self,
        question: str,
        index_filenames: Optional[List[str]] = None,
        use_deep_traversal: Optional[bool] = None,
        max_depth: Optional[int] = None,
        max_branches: Optional[int] = None,
        domain_template: str = "general",
        language: str = "ko",
        node_context: Optional[NodeContext] = None,
        enable_comparison: bool = False
    ) -> ChatResult:
        if not question or not question.strip():
            return ChatResult(
                success=False,
                error_message="Question cannot be empty"
            )
        
        routing_result = self.document_router.route(question, index_filenames)
        
        if not routing_result.selected_indices:
            return ChatResult(
                success=False,
                error_message="No indexed documents found. Please upload and index documents first."
            )
        
        for idx_file in routing_result.selected_indices:
            if not idx_file.endswith("_index.json"):
                return ChatResult(
                    success=False,
                    error_message=f"Invalid index filename format: {idx_file}"
                )
        
        use_traversal = use_deep_traversal if use_deep_traversal is not None else Config.USE_DEEP_TRAVERSAL
        depth = max_depth if max_depth is not None else Config.MAX_TRAVERSAL_DEPTH
        branches = max_branches if max_branches is not None else Config.MAX_BRANCHES_PER_LEVEL
        
        try:
            reasoner = TreeRAGReasoner(
                routing_result.selected_indices,
                use_deep_traversal=use_traversal
            )
            
            actual_question = self._enhance_question(question, node_context)
            
            answer, traversal_info = reasoner.query(
                actual_question,
                enable_comparison=enable_comparison,
                max_depth=depth,
                max_branches=branches,
                domain_template=domain_template,
                language=language
            )
            
            citations = self._extract_citations(answer)
            
            comparison = None
            if len(routing_result.selected_indices) > 1 and enable_comparison:
                comparison = self._extract_comparison(answer, routing_result.selected_indices)
            
            trav_info = TraversalInfo(
                used_deep_traversal=traversal_info["used_deep_traversal"],
                nodes_visited=traversal_info["nodes_visited"],
                nodes_selected=traversal_info["nodes_selected"],
                max_depth=traversal_info["max_depth"],
                max_branches=traversal_info["max_branches"]
            )
            
            resolved_refs = None
            if "resolved_references" in traversal_info:
                resolved_refs = [
                    ResolvedReference(**ref)
                    for ref in traversal_info["resolved_references"]
                ]
            
            hallucination_warning = self._detect_hallucination(
                answer, 
                traversal_info["nodes_selected"]
            )
            
            return ChatResult(
                success=True,
                answer=answer,
                citations=citations,
                comparison=comparison,
                traversal_info=trav_info,
                resolved_references=resolved_refs,
                hallucination_warning=hallucination_warning
            )
        
        except Exception as e:
            return ChatResult(
                success=False,
                error_message=f"Chat processing failed: {str(e)}"
            )
    
    def _enhance_question(
        self, 
        question: str, 
        node_context: Optional[NodeContext]
    ) -> str:
        if not node_context:
            return question
        
        page_info = f" (페이지: {node_context.page_ref})" if node_context.page_ref else ""
        
        return f"""[컨텍스트: 문서 섹션 "{node_context.title}"]

사용자가 위 섹션에 대해 질문하고 있습니다.{page_info}

질문: {question}

이 섹션과 관련된 내용을 중심으로 상세히 답변해주세요."""
    
    def _extract_citations(self, answer: str) -> List[str]:
        """답변에서 인용 추출
        
        Args:
            answer: LLM 답변
            
        Returns:
            List[str]: 인용 목록
        """
        patterns = [
            r'\[출처:[^\]]+\]',
            r'\(출처:[^)]+\)',
            r'(?:문서|섹션):\s*[^,\n]+,\s*p\.?\s*\d+',
            r'「[^」]+」,?\s*p\.?\s*\d+',
            r'『[^』]+』,?\s*p\.?\s*\d+',
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                clean = match.strip('[]()「」『』')
                clean = clean.replace('출처:', '').strip()
                if clean and clean not in citations:
                    citations.append(clean)
        
        page_pattern = r'([가-힣a-zA-Z0-9_\-\s]+),\s*p\.?\s*(\d+(?:-\d+)?)'
        page_matches = re.findall(page_pattern, answer)
        for doc, page in page_matches:
            citation = f"{doc.strip()}, p.{page}"
            if citation not in citations:
                citations.append(citation)
        
        return citations
    
    def _extract_comparison(
        self, 
        answer: str, 
        selected_indices: List[str]
    ) -> Optional[ComparisonResult]:
        comparison_keywords = ['공통점', '차이점', '비교', '반면', '대조적으로', '유사하게']
        
        has_comparison = any(kw in answer for kw in comparison_keywords)
        
        if not has_comparison:
            return None
        
        doc_names = [idx.replace('_index.json', '') for idx in selected_indices]
        
        commonalities = None
        differences = None
        
        common_patterns = [
            r'공통점[:\s]*([^차이점]+?)(?=차이점|$)',
            r'공통적으로[:\s]*([^.]+\.)',
        ]
        for pattern in common_patterns:
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                commonalities = match.group(1).strip()
                break
        
        diff_patterns = [
            r'차이점[:\s]*(.+?)(?=공통점|$)',
            r'다른 점[:\s]*(.+?)(?=같은|$)',
        ]
        for pattern in diff_patterns:
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                differences = match.group(1).strip()
                break
        
        return ComparisonResult(
            has_comparison=True,
            documents_compared=doc_names,
            commonalities=commonalities,
            differences=differences
        )
    
    def _detect_hallucination(
        self,
        answer: str,
        nodes_selected: List[Dict[str, Any]]
    ) -> Optional[HallucinationWarning]:
        detector = HallucinationDetector(confidence_threshold=self.hallucination_threshold)
        result = detector.detect(answer, nodes_selected)
        
        if result["is_reliable"]:
            return None
        
        if len(result["sentence_analysis"]) == 0:
            return None
        
        low_conf_count = sum(
            1 for s in result["sentence_analysis"] 
            if not s["is_grounded"]
        )
        total_count = len(result["sentence_analysis"])
        low_conf_ratio = low_conf_count / total_count if total_count > 0 else 0
        
        if low_conf_ratio >= 0.7:
            print(f"Hallucination detected: {low_conf_count}/{total_count} sentences low confidence")
            return HallucinationWarning(
                message=f"{low_conf_count}/{total_count} sentences have low confidence",
                overall_confidence=result["overall_confidence"],
                threshold=self.hallucination_threshold
            )
        
        return None
