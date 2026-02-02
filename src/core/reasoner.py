import json
import os
from typing import Any, List, Dict, Optional
from src.config import Config
from src.core.tree_traversal import TreeNavigator, format_traversal_results
from src.core.reference_resolver import ReferenceResolver
from src.utils.cache import get_cache
from src.utils.hallucination_detector import create_detector

DOMAIN_PROMPTS = {
    "general": """당신은 전문 문서 분석 AI 어시스턴트입니다.
제공된 문서의 인덱스를 사용하여 사용자의 질문에 정확하게 답변하세요.""",
    
    "medical": """당신은 의료 전문 AI 어시스턴트입니다.
**의료 문서 분석 원칙:**
- 의학 용어를 정확하게 사용하고 필요시 설명을 추가하세요
- 임상 가이드라인과 근거 기반 의학(EBM)을 준수하세요
- 진단, 치료, 약물에 대한 정보는 반드시 페이지 참조와 함께 제공하세요
- 부작용, 금기사항, 주의사항을 명확히 명시하세요
- 불확실한 정보는 "추가 확인이 필요합니다"라고 명시하세요""",
    
    "legal": """당신은 법률 전문 AI 어시스턴트입니다.
**법률 문서 분석 원칙:**
- 법조문과 조항을 정확히 인용하고 페이지 번호를 명시하세요
- 조건, 예외사항, 단서조항을 빠짐없이 포함하세요
- "~할 수 있다", "~하여야 한다" 등의 법률 용어를 정확히 사용하세요
- 판례나 선례가 언급된 경우 명확히 표시하세요
- 법적 해석이 필요한 부분은 여러 관점을 제시하세요""",
    
    "financial": """당신은 재무/금융 전문 AI 어시스턴트입니다.
**재무 문서 분석 원칙:**
- 숫자, 지표, 통계는 절대적으로 정확해야 하며 반드시 출처를 명시하세요
- 재무제표 항목(자산, 부채, 수익 등)을 정확히 구분하세요
- 회계 기준(K-IFRS, GAAP 등)이 명시된 경우 이를 고려하세요
- 전년 대비 증감률, 비율 등을 제시할 때 계산 근거를 설명하세요
- 리스크 요인, 우발채무 등 주요 재무 위험을 명확히 표시하세요""",
    
    "academic": """당신은 학술 연구 전문 AI 어시스턴트입니다.
**학술 문서 분석 원칙:**
- 연구 방법론, 실험 설계, 데이터 분석 방법을 명확히 구분하세요
- 연구 결과와 저자의 해석/주장을 구분하여 제시하세요
- 통계적 유의성(p-value), 신뢰구간 등 정량적 지표를 정확히 인용하세요
- 선행연구와의 관계, 연구의 한계점을 명시하세요
- 인용 형식을 정확히 따르고 페이지 번호를 반드시 포함하세요"""
}

LANGUAGE_INSTRUCTIONS = {
    "ko": "**중요: 모든 답변은 반드시 한국어로 작성하세요.**",
    "en": "**IMPORTANT: You MUST respond in English only.**",
    "ja": "**重要：必ず日本語で回答してください。**"
}

class TreeRAGReasoner:
    def __init__(self, index_filenames: List[str], use_deep_traversal: bool = True):
        self.index_trees: List[Dict[str, Any]] = []
        self.index_filenames = index_filenames
        self.use_deep_traversal = use_deep_traversal
        
        for index_filename in index_filenames:
            path = os.path.join(Config.INDEX_DIR, index_filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.index_trees.append(json.load(f))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in index file {index_filename}: {e}")
            except IOError as e:
                raise IOError(f"Failed to read index file {index_filename}: {e}")


    def query(self, user_question: str, enable_comparison: bool = True, max_depth: int = 5, max_branches: int = 3, domain_template: str = "general", language: str = "ko", node_context: Optional[dict] = None) -> tuple[str, dict]:
        if not user_question or not user_question.strip():
            raise ValueError("user_question cannot be empty")
        
        cache = get_cache()
        cached_response = cache.get(
            question=user_question,
            index_files=self.index_filenames,
            use_deep_traversal=self.use_deep_traversal,
            max_depth=max_depth,
            max_branches=max_branches,
            domain_template=domain_template,
            language=language,
            node_context=node_context
        )
        
        if cached_response:
            print(f"✅ Cache HIT - Returning cached response")
            return cached_response["answer"], cached_response["metadata"]
        
        print(f"❌ Cache MISS - Generating new response")
        
        traversal_info = {
            "used_deep_traversal": self.use_deep_traversal,
            "nodes_visited": [],
            "nodes_selected": [],
            "max_depth": max_depth,
            "max_branches": max_branches
        }
        
        reference_context = ""
        resolved_refs = []
        for tree in self.index_trees:
            resolver = ReferenceResolver(tree)
            refs = resolver.detect_references(user_question)
            if refs:
                resolved_nodes = resolver.resolve_all_references(user_question)
                if resolved_nodes:
                    resolved_refs.extend(resolved_nodes)
                    ref_context = resolver.format_resolved_context(resolved_nodes)
                    if ref_context:
                        reference_context += ref_context
                        print(f"📎 Resolved {len(resolved_nodes)} cross-references: {[r.get('title') for r in resolved_nodes]}")
        
        if self.use_deep_traversal:
            print("🌲 Using deep tree traversal mode")
            context_str, trav_data = self._build_context_with_traversal(user_question, max_depth, max_branches)
            traversal_info.update(trav_data)
        else:
            print("📄 Using flat context mode (legacy)")
            context_str = self._build_flat_context()
        
        if reference_context:
            context_str = reference_context + "\n\n" + context_str
        
        is_multi_doc = len(self.index_filenames) > 1
        comparison_prompt = ""
        
        if is_multi_doc and enable_comparison:
            comparison_prompt = f"""

### 📊 다중 문서 비교 분석 (필수):
여러 문서가 제공되었으므로, 반드시 다음 형식으로 비교 분석을 포함하세요:

**1. 공통점 (Commonalities)**
- 모든 문서에서 일치하는 내용
- 예: "모든 교육과정에서 졸업 학점은 130학점 이상 [문서A, p.5], [문서B, p.3]"

**2. 차이점 (Differences)**
표 형식으로 명확히 구분:
| 항목 | {self.index_filenames[0].replace('_index.json', '')} | {self.index_filenames[1].replace('_index.json', '') if len(self.index_filenames) > 1 else '기타'} |
|------|------|------|
| 예: 필수학점 | 18학점 [p.5] | 21학점 [p.4] |
| 예: 선택과목 | 10개 [p.7] | 15개 [p.6] |

**3. 문서 우선순위 (해당시)**
- 충돌하는 내용이 있다면, 어떤 문서가 최신/공식인지 명시
- 예: "최신 버전(2024)의 내용이 적용됩니다 [문서A, p.10]"
"""

        domain_prompt = DOMAIN_PROMPTS.get(domain_template, DOMAIN_PROMPTS["general"])
        
        language_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["ko"])
        
        prompt = f"""
{domain_prompt}

{language_instruction}

### 📋 답변 작성 단계 (반드시 순서대로):

**STEP 1: 질문 핵심 파악**
- 질문에서 요구하는 핵심 정보가 무엇인지 명확히 파악
- 예: "졸업 학점은?" → 숫자(학점) 찾기, "필수 과목은?" → 과목명 리스트 찾기

**STEP 2: 인덱스에서 정확한 정보 검색**
- 제공된 인덱스 JSON에서 질문과 관련된 섹션 찾기
- page_ref, title, summary 필드를 활용하여 정확한 위치 특정

**STEP 3: 핵심 답변 먼저 작성 (1-2문장)**
- 질문에 대한 직접적인 답변을 먼저 명확하게 제시
- 반드시 페이지 참조 포함: [문서명, p.페이지]
- 예: "졸업 학점은 130학점입니다 [인공지능반도체, p.2]."

**STEP 4: 상세 설명 추가 (필요시)**
- 핵심 답변 이후 추가 맥락이나 상세 정보 제공
- 모든 문장에 페이지 참조 포함

**STEP 5: 참조 페이지 요약**
- 답변 마지막에 📚 **참조 페이지** 형식으로 모든 출처 나열

### ⚠️ 중요 규칙:

1. **인덱스에 없는 정보는 절대 추측하지 마세요** - "인덱스에 해당 정보가 명시되어 있지 않습니다"라고 답변
2. **페이지 번호 필수** - 모든 사실적 진술에 [문서명, p.번호] 형식으로 표기
3. **간결하고 정확하게** - 질문에 직접 답하는 정보를 우선 제시
4. **숫자/이름은 정확히** - 학점 수, 과목명, 날짜 등은 인덱스에 있는 그대로 기재
{comparison_prompt}

### 답변 템플릿:

[핵심 답변 1-2문장 + 페이지 참조]

[상세 설명 (필요시) + 페이지 참조]
{f"\n[문서 비교 분석: 공통점/차이점 표]" if is_multi_doc else ""}

📚 **참조 페이지**: [문서명, p.X], [문서명, p.Y-Z]

### 컨텍스트:
{context_str}

### 질문:
{user_question}

### 답변 (위 규칙을 철저히 따라 작성):
"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt
            )
            if not response.text:
                raise ValueError("Empty response from model")
            
            if resolved_refs:
                traversal_info["resolved_references"] = [
                    {
                        "title": ref.get("title", ""),
                        "page_ref": ref.get("page_ref"),
                        "summary": ref.get("summary")
                    }
                    for ref in resolved_refs
                ]
            
            detector = create_detector(confidence_threshold=0.6)
            
            source_nodes = []
            if self.use_deep_traversal:
                for tree_idx, tree in enumerate(self.index_trees):
                    doc_name = self.index_filenames[tree_idx].replace("_index.json", "")
                    navigator = TreeNavigator(tree, doc_name)
                    relevant_nodes, _ = navigator.search(
                        query=user_question,
                        max_depth=max_depth,
                        max_branches=max_branches
                    )
                    source_nodes.extend([node["node"] for node in relevant_nodes])
            else:
                for tree in self.index_trees:
                    source_nodes.extend(self._extract_all_nodes(tree))
            
            if resolved_refs:
                source_nodes.extend(resolved_refs)
            
            detection_result = detector.detect(response.text, source_nodes)
            
            traversal_info["hallucination_detection"] = {
                "overall_confidence": detection_result["overall_confidence"],
                "is_reliable": detection_result["is_reliable"],
                "hallucinated_count": detection_result["hallucinated_count"],
                "total_sentences": detection_result["total_sentences"]
            }
            
            if detection_result["is_reliable"]:
                print(f"✅ Hallucination check: {detection_result['overall_confidence']:.1%} confidence")
            else:
                print(f"⚠️ Hallucination detected: {detection_result['hallucinated_count']}/{detection_result['total_sentences']} sentences low confidence")
            

            cache = get_cache()
            cache_data = {
                "answer": response.text,
                "metadata": traversal_info
            }
            cache.set(
                question=user_question,
                index_files=self.index_filenames,
                use_deep_traversal=self.use_deep_traversal,
                max_depth=max_depth,
                max_branches=max_branches,
                domain_template=domain_template,
                language=language,
                response=cache_data,
                node_context=node_context
            )
            print(f"💾 Response cached")
            
            return response.text, traversal_info
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    def _build_context_with_traversal(self, query: str, max_depth: int, max_branches: int) -> tuple[str, dict]:
        all_results = []
        all_visited = []
        all_selected = []
        
        for idx, tree in enumerate(self.index_trees):
            doc_name = self.index_filenames[idx].replace("_index.json", "")
            navigator = TreeNavigator(tree, doc_name)
            relevant_nodes, trav_stats = navigator.search(
                query=query,
                max_depth=max_depth,
                max_branches=max_branches
            )
            formatted = format_traversal_results(relevant_nodes, doc_name)
            all_results.append(formatted)
            
            all_visited.extend([f"{doc_name}: {title}" for title in trav_stats["visited_titles"]])
            all_selected.extend([{
                "document": doc_name,
                "title": node["node"].get("title", "Untitled"),
                "page_ref": node["node"].get("page_ref", "N/A")
            } for node in relevant_nodes])
        
        traversal_data = {
            "nodes_visited": all_visited,
            "nodes_selected": all_selected
        }
        
        return "\n\n---\n\n".join(all_results), traversal_data
    
    def _build_flat_context(self) -> str:
        combined_context = []
        for idx, tree in enumerate(self.index_trees):
            doc_name = self.index_filenames[idx].replace("_index.json", "")
            combined_context.append({
                "document": doc_name,
                "content": tree
            })
        
        return json.dumps(combined_context, ensure_ascii=False)
    
    def _extract_all_nodes(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all nodes from a tree structure recursively.
        
        Args:
            tree: Tree structure (dict with 'children' list)
        
        Returns:
            Flat list of all nodes
        """
        nodes = []
        
        def traverse(node):
            if isinstance(node, dict):
                nodes.append(node)
                if "children" in node and isinstance(node["children"], list):
                    for child in node["children"]:
                        traverse(child)
        
        traverse(tree)
        return nodes