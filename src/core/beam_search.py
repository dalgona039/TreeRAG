
import json
import heapq
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from src.config import Config


@dataclass(order=True)
class BeamNode:
    priority: float  
    node: Dict[str, Any] = field(compare=False)
    path: str = field(compare=False)
    depth: int = field(compare=False)
    cumulative_score: float = field(compare=False)
    parent_context: str = field(compare=False)
    
    @classmethod
    def create(
        cls,
        node: Dict[str, Any],
        path: str,
        depth: int,
        score: float,
        parent_context: str = ""
    ) -> "BeamNode":
        return cls(
            priority=-score,  
            node=node,
            path=path,
            depth=depth,
            cumulative_score=score,
            parent_context=parent_context
        )


@dataclass
class RelevanceScore:
    is_relevant: bool
    score: float 
    reason: str
    

class BeamSearchNavigator:
    MAX_DEPTH_LIMIT = 10
    MAX_NODES_LIMIT = 500
    
    SEMANTIC_WEIGHT = 0.6    
    KEYWORD_WEIGHT = 0.2     
    STRUCTURE_WEIGHT = 0.2   
    
    def __init__(
        self, 
        tree: Dict[str, Any], 
        document_name: str,
        beam_width: int = 5
    ):
        self.tree = tree
        self.document_name = document_name
        self.beam_width = beam_width
        
        self.visited_nodes: List[str] = []
        self.selected_nodes: List[Dict[str, Any]] = []
        self.node_scores: Dict[str, float] = {}  # node_id -> score
        
        self.total_evaluated = 0
        self.total_expanded = 0
    
    def search(
        self,
        query: str,
        max_depth: int = 5,
        min_score_threshold: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        print(f"  [Beam Search] Starting traversal")
        print(f"   Document: {self.document_name}")
        print(f"   Beam Width: {self.beam_width}")
        print(f"   Max Depth: {max_depth}")
        print(f"   Query: {query[:80]}...")
        
        self.visited_nodes = []
        self.selected_nodes = []
        self.node_scores = {}
        self.total_evaluated = 0
        self.total_expanded = 0
        
        root = self.tree
        root_beam = BeamNode.create(
            node=root,
            path=root.get("title", "Root"),
            depth=0,
            score=1.0, 
            parent_context=""
        )
        
        current_beam: List[BeamNode] = [root_beam]
        
        for depth in range(max_depth):
            if not current_beam:
                break
            
            print(f"\n     Level {depth}: {len(current_beam)} nodes in beam")
            
            all_children: List[Tuple[BeamNode, Dict[str, Any]]] = []
            
            for beam_node in current_beam:
                node = beam_node.node
                node_id = node.get("id", str(id(node)))
                
                if node_id in self.visited_nodes:
                    continue
                self.visited_nodes.append(node_id)
                
                children = node.get("children", [])
                
                if not children or depth == max_depth - 1:
                    if beam_node.cumulative_score >= min_score_threshold:
                        self.selected_nodes.append({
                            "node": node,
                            "path": beam_node.path,
                            "depth": depth,
                            "score": beam_node.cumulative_score
                        })
                        print(f"      ✓ Selected: {node.get('title', 'Untitled')[:40]} (score: {beam_node.cumulative_score:.2f})")
                    continue
                
                for child in children:
                    all_children.append((beam_node, child))
                
                self.total_expanded += 1
            
            if not all_children:
                break
            
            scored_children = self._batch_score_nodes(
                all_children, 
                query, 
                depth + 1
            )
            
            scored_children.sort(key=lambda x: x.cumulative_score, reverse=True)
            current_beam = scored_children[:self.beam_width]
            
            for beam_node in current_beam:
                print(f"      → Beam: {beam_node.node.get('title', 'Untitled')[:40]} (score: {beam_node.cumulative_score:.2f})")
        
        for beam_node in current_beam:
            node = beam_node.node
            node_id = node.get("id", str(id(node)))
            
            already_selected = any(
                s["node"].get("id") == node_id 
                for s in self.selected_nodes
            )
            
            if not already_selected and beam_node.cumulative_score >= min_score_threshold:
                self.selected_nodes.append({
                    "node": node,
                    "path": beam_node.path,
                    "depth": beam_node.depth,
                    "score": beam_node.cumulative_score
                })
        
        self.selected_nodes.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        stats = {
            "algorithm": "beam_search",
            "beam_width": self.beam_width,
            "nodes_visited": len(self.visited_nodes),
            "nodes_evaluated": self.total_evaluated,
            "nodes_expanded": self.total_expanded,
            "nodes_selected": len(self.selected_nodes),
            "max_depth_used": max_depth
        }
        
        print(f"\n   ✅ Beam Search complete: {len(self.selected_nodes)} nodes selected")
        print(f"      Evaluated: {self.total_evaluated}, Expanded: {self.total_expanded}")
        
        return self.selected_nodes, stats
    
    def _batch_score_nodes(
        self,
        children_with_parents: List[Tuple[BeamNode, Dict[str, Any]]],
        query: str,
        depth: int
    ) -> List[BeamNode]:
        if not children_with_parents:
            return []
        
        node_infos = []
        for parent, child in children_with_parents:
            node_infos.append({
                "index": len(node_infos),
                "title": child.get("title", ""),
                "summary": child.get("summary", "")[:300],
                "page_ref": child.get("page_ref", ""),
                "parent_path": parent.path
            })
        
        self.total_evaluated += len(node_infos)
        
        scores = self._batch_llm_score(node_infos, query)
        
        result = []
        for i, (parent, child) in enumerate(children_with_parents):
            llm_score = scores.get(i, 0.5)
            
            keyword_score = self._keyword_score(child, query)
            
            structure_score = max(0.3, 1.0 - (depth * 0.1))
            
            combined_score = (
                self.SEMANTIC_WEIGHT * llm_score +
                self.KEYWORD_WEIGHT * keyword_score +
                self.STRUCTURE_WEIGHT * structure_score
            )
            
            cumulative_score = parent.cumulative_score * combined_score
            
            new_path = f"{parent.path} > {child.get('title', 'Untitled')}"
            
            result.append(BeamNode.create(
                node=child,
                path=new_path,
                depth=depth,
                score=cumulative_score,
                parent_context=parent.path
            ))
            
            self.node_scores[child.get("id", str(id(child)))] = cumulative_score
        
        return result
    
    def _batch_llm_score(
        self,
        node_infos: List[Dict[str, Any]],
        query: str
    ) -> Dict[int, float]:
        """배치 LLM 점수 계산
        
        여러 노드를 한 번의 LLM 호출로 평가
        
        Args:
            node_infos: 노드 정보 목록
            query: 사용자 질문
            
        Returns:
            Dict[노드 인덱스, 점수]
        """
        if len(node_infos) == 0:
            return {}

        # Quota-saver: skip LLM node scoring entirely (keyword + structure only).
        # Cuts a TreeRAG query from ~6 Gemini calls to 1 (final answer). Opt-in
        # via env so default behaviour is unchanged.
        import os as _os
        if _os.getenv("TREERAG_DISABLE_LLM_SCORING", "").strip().lower() in ("1", "true", "yes", "on"):
            return {}

        # 배치 크기 제한 (너무 크면 토큰 초과)
        MAX_BATCH = 10
        
        if len(node_infos) > MAX_BATCH:
            # 분할 처리
            results = {}
            for i in range(0, len(node_infos), MAX_BATCH):
                batch = node_infos[i:i + MAX_BATCH]
                batch_results = self._batch_llm_score(batch, query)
                for idx, score in batch_results.items():
                    results[i + idx] = score
            return results
        
        # 프롬프트 생성
        nodes_text = json.dumps(node_infos, ensure_ascii=False, indent=2)
        
        prompt = f"""당신은 문서 검색 관련성 평가 전문가입니다.
사용자 질문에 대해 각 섹션의 관련성 점수(0.0~1.0)를 평가하세요.

### 사용자 질문:
{query}

### 평가 대상 섹션들:
{nodes_text}

### 평가 기준:
- 1.0: 질문에 직접 답변 가능
- 0.7-0.9: 매우 관련성 높음
- 0.4-0.6: 어느 정도 관련 있음
- 0.1-0.3: 약간 관련 있음
- 0.0: 전혀 관련 없음

### 응답 형식 (JSON):
{{
  "scores": [
    {{"index": 0, "score": 0.8, "reason": "이유"}},
    {{"index": 1, "score": 0.3, "reason": "이유"}},
    ...
  ]
}}

JSON만 출력하세요:
"""
        
        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                return {i: 0.5 for i in range(len(node_infos))}
            
            result = json.loads(response.text)
            scores_list = result.get("scores", [])
            
            return {
                item["index"]: item.get("score", 0.5)
                for item in scores_list
                if "index" in item
            }
            
        except Exception as e:
            print(f"      ⚠️ Batch scoring failed: {e}")
            return {i: 0.5 for i in range(len(node_infos))}
    
    def _keyword_score(self, node: Dict[str, Any], query: str) -> float:
        """키워드 매칭 점수 계산
        
        Args:
            node: 노드
            query: 사용자 질문
            
        Returns:
            0.0 ~ 1.0 점수
        """
        title = node.get("title", "").lower()
        summary = node.get("summary", "").lower()
        node_text = f"{title} {summary}"
        
        # 쿼리 토큰화 (간단한 공백 분리)
        query_tokens = set(query.lower().split())
        
        # 불용어 제거
        stopwords = {"을", "를", "이", "가", "은", "는", "의", "에", "로", "와", "과", "에서", "the", "a", "an", "is", "are", "was", "were"}
        query_tokens = query_tokens - stopwords
        
        if not query_tokens:
            return 0.5
        
        # 매칭 비율
        matched = sum(1 for token in query_tokens if token in node_text)
        score = matched / len(query_tokens)
        
        return min(1.0, score)


def format_beam_results(results: List[Dict[str, Any]], document_name: str) -> str:
    """Beam Search 결과 포맷팅"""
    if not results:
        return f"No relevant sections found in {document_name}."
    
    output = [f"### 📄 {document_name} (발견: {len(results)}개, Beam Search)\n"]
    
    for idx, item in enumerate(results, 1):
        node = item["node"]
        path = item["path"]
        depth = item["depth"]
        score = item.get("score", 0.0)
        
        output.append(f"\n**[{idx}] {path}** (점수: {score:.2f})")
        output.append(f"- 페이지: {node.get('page_ref', 'N/A')}")
        output.append(f"- 깊이: {depth}")
        if node.get("summary"):
            output.append(f"- 요약: {node['summary']}")
        if node.get("text"):
            output.append(f"- 원문:\n{node['text'][:500]}...")
        output.append("")
    
    return "\n".join(output)
