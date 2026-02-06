import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import HTTPException
from src.config import Config


class TreeNavigator:
    MAX_DEPTH_LIMIT = 10
    MAX_NODES_LIMIT = 1000
    
    def __init__(self, tree: Dict[str, Any], document_name: str):
        self.tree = tree
        self.document_name = document_name
        self.visited_nodes: List[str] = []
        self.relevant_nodes: List[Dict[str, Any]] = []
        self.visited_titles: List[str] = []
        self.node_count = 0
    
    def search(
        self, 
        query: str, 
        max_depth: int = 5,
        max_branches: int = 3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        print(f"🔍 Starting deep traversal for: {self.document_name}")
        print(f"   Query: {query[:100]}...")
        
        if max_depth > self.MAX_DEPTH_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"max_depth ({max_depth}) exceeds limit ({self.MAX_DEPTH_LIMIT}). "
                       f"Document structure is too complex for deep traversal."
            )
        
        self.node_count = 0
        self.visited_nodes = []
        self.relevant_nodes = []
        self.visited_titles = []
        
        root = self.tree
        try:
            self._traverse_iterative(root, query, max_depth, max_branches)
        except HTTPException:
            raise
        except Exception as e:
            print(f"⚠️ Unexpected error during traversal: {e}")
            raise HTTPException(status_code=500, detail=f"Traversal failed: {str(e)}")
        
        print(f"✅ Found {len(self.relevant_nodes)} relevant sections")
        
        traversal_stats = {
            "nodes_visited": len(self.visited_nodes),
            "visited_titles": self.visited_titles[:],
            "nodes_selected": len(self.relevant_nodes),
            "max_depth_used": max_depth,
            "max_branches_used": max_branches
        }
        
        return self.relevant_nodes, traversal_stats
    
    def _traverse_iterative(
        self,
        root: Dict[str, Any],
        query: str,
        max_depth: int,
        max_branches: int
    ) -> None:
        """Iterative DFS using stack to avoid recursion limits."""
        stack = [(root, 0, "")]
        
        while stack:
            node, current_depth, parent_context = stack.pop()
            
            node_id = node.get("id", "unknown")
            node_title = node.get("title", "Untitled")
            
            self.node_count += 1
            if self.node_count > self.MAX_NODES_LIMIT:
                raise HTTPException(
                    status_code=413,
                    detail=f"Document too large: visited {self.node_count} nodes. "
                           f"Please use a more specific query or reduce max_depth."
                )
            
            if current_depth >= max_depth:
                continue
            
            if node_id in self.visited_nodes:
                continue
            
            self.visited_nodes.append(node_id)
            self.visited_titles.append(node_title)
            
            is_relevant = self._evaluate_node_relevance(node, query, parent_context, current_depth)
            
            if is_relevant:
                if not node.get("children") or current_depth >= max_depth - 1:
                    self.relevant_nodes.append({
                        "node": node,
                        "path": parent_context + f" > {node_title}" if parent_context else node_title,
                        "depth": current_depth
                    })
                    print(f"   ✓ Depth {current_depth}: {node_title}")
                
                if current_depth < max_depth and node.get("children"):
                    children = node["children"]
                    
                    if len(children) > max_branches:
                        children = self._select_most_relevant_children(
                            children, 
                            query, 
                            max_branches,
                            parent_context + f" > {node_title}" if parent_context else node_title
                        )
                    
                    new_context = parent_context + f" > {node_title}" if parent_context else node_title
                    for child in reversed(children):
                        if len(stack) + self.node_count > self.MAX_NODES_LIMIT:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Document structure too complex: stack size + visited nodes exceeds limit. "
                                       f"Please use a more specific query or reduce max_depth/max_branches."
                            )
                        stack.append((child, current_depth + 1, new_context))
    
    def _evaluate_node_relevance(
        self,
        node: Dict[str, Any],
        query: str,
        parent_context: str,
        depth: int
    ) -> bool:
        if depth == 0:
            return True
        
        title = node.get("title", "")
        summary = node.get("summary", "")
        page_ref = node.get("page_ref", "")
        
        if len(summary) < 20 and len(title) < 10:
            return False
        
        prompt = f"""
당신은 문서 탐색 전문가입니다.
사용자의 질문에 답하기 위해, 아래 섹션이 관련이 있는지 판단하세요.

### 컨텍스트 (문서 경로):
{parent_context}

### 평가 대상 섹션:
- 제목: {title}
- 요약: {summary}
- 페이지: {page_ref}

### 사용자 질문:
{query}

### 판단 기준:
1. 이 섹션이 질문에 직접 답할 수 있는가?
2. 이 섹션의 하위 항목에 답이 있을 가능성이 높은가?
3. 키워드나 개념이 겹치는가?

답변 형식 (JSON):
{{
  "relevant": true 또는 false,
  "reason": "1-2문장 설명"
}}

JSON만 출력하세요:
"""
        
        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            if not response.text:
                print(f"   ⚠️ No response for {title}, marking as not relevant")
                return False
            
            result = json.loads(response.text)
            is_relevant = result.get("relevant", False)
            
            if is_relevant:
                print(f"   → Exploring: {title} (Reason: {result.get('reason', 'N/A')[:50]}...)")
            
            return is_relevant
            
        except Exception as e:
            print(f"⚠️ Relevance check failed: {e}")
            query_lower = query.lower()
            title_lower = title.lower()
            return any(keyword in title_lower for keyword in query_lower.split()[:5])
    
    def _select_most_relevant_children(
        self,
        children: List[Dict[str, Any]],
        query: str,
        max_branches: int,
        parent_context: str
    ) -> List[Dict[str, Any]]:
        children_summaries = []
        for idx, child in enumerate(children):
            children_summaries.append({
                "index": idx,
                "title": child.get("title", ""),
                "summary": child.get("summary", "")[:200],
                "page_ref": child.get("page_ref", "")
            })
        
        prompt = f"""
당신은 문서 탐색 전문가입니다.
다음 하위 섹션들 중에서, 사용자 질문에 답하기 위해 우선적으로 탐색해야 할 {max_branches}개를 선택하세요.

### 상위 컨텍스트:
{parent_context}

### 하위 섹션 목록:
{json.dumps(children_summaries, ensure_ascii=False, indent=2)}

### 사용자 질문:
{query}

### 선택 기준:
- 질문과의 직접적 관련성
- 답변에 필요한 정보가 있을 가능성
- 우선순위 (중요도 높은 순)

답변 형식 (JSON):
{{
  "selected_indices": [0, 3, 5],
  "reason": "선택 이유 1-2문장"
}}

최대 {max_branches}개의 인덱스를 선택하세요. JSON만 출력:
"""
        
        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            if not response.text:
                print(f"   ⚠️ No response for child selection, selecting first {max_branches}")
                return children[:max_branches]
            
            result = json.loads(response.text)
            selected_indices = result.get("selected_indices", [])
            selected_indices = [i for i in selected_indices if 0 <= i < len(children)]
            
            print(f"   📊 Selected {len(selected_indices)}/{len(children)} branches: {result.get('reason', '')[:60]}...")
            
            return [children[i] for i in selected_indices]
            
        except Exception as e:
            print(f"⚠️ Branch selection failed: {e}, using first {max_branches}")
            return children[:max_branches]


def format_traversal_results(results: List[Dict[str, Any]], document_name: str) -> str:
    if not results:
        return f"No relevant sections found in {document_name}."
    
    output = [f"### 📄 {document_name} (발견된 관련 섹션: {len(results)}개)\n"]
    
    for idx, item in enumerate(results, 1):
        node = item["node"]
        path = item["path"]
        depth = item["depth"]
        
        output.append(f"\n**[{idx}] {path}**")
        output.append(f"- 페이지: {node.get('page_ref', 'N/A')}")
        output.append(f"- 깊이: {depth}")
        if node.get("summary"):
            output.append(f"- 요약: {node['summary']}")
        if node.get("text"):
            output.append(f"- 원문:\n{node['text']}")
        output.append("")
    
    return "\n".join(output)
