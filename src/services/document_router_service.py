
import json
from typing import List, Optional
from dataclasses import dataclass

from src.config import Config
from src.repositories import IndexRepository


@dataclass
class RoutingResult:
    selected_indices: List[str]
    total_available: int
    routing_method: str 


class DocumentRouterService:
    def __init__(self, index_repository: Optional[IndexRepository] = None):
        self.index_repo = index_repository or IndexRepository()
    
    def route(
        self,
        question: str,
        user_specified_indices: Optional[List[str]] = None
    ) -> RoutingResult:
        available_indices = self.index_repo.list_all()
        
        if user_specified_indices and len(user_specified_indices) > 0:
            valid_indices = [
                idx for idx in user_specified_indices 
                if idx in available_indices
            ]
            return RoutingResult(
                selected_indices=valid_indices if valid_indices else available_indices,
                total_available=len(available_indices),
                routing_method="user_specified"
            )
        
        if not available_indices:
            return RoutingResult(
                selected_indices=[],
                total_available=0,
                routing_method="all"
            )
        
        if len(available_indices) == 1:
            return RoutingResult(
                selected_indices=available_indices,
                total_available=1,
                routing_method="all"
            )
        
        selected = self._auto_route(question, available_indices)
        
        return RoutingResult(
            selected_indices=selected,
            total_available=len(available_indices),
            routing_method="auto_selected"
        )
    
    def _auto_route(self, question: str, available_indices: List[str]) -> List[str]:
        doc_summaries = []
        for filename in available_indices:
            summary = self.index_repo.get_summary(filename)
            doc_name = filename.replace("_index.json", "")
            if summary:
                doc_summaries.append(f"- {doc_name}: {summary[:200]}")
            else:
                doc_summaries.append(f"- {doc_name}")
        
        context = "\n".join(doc_summaries)
        
        prompt = f"""당신은 문서 라우터입니다.
사용자의 질문을 분석하여, 어떤 규제 문서를 참조해야 하는지 선택하세요.

### 사용 가능한 문서:
{context}

### 사용자 질문:
{question}

### 규칙:
1. 질문과 가장 관련 있는 문서를 선택하세요.
2. 여러 문서가 관련되어 있다면 모두 선택하세요.
3. 반드시 위 목록에 있는 문서명만 사용하세요.
4. 응답 형식: JSON 배열로만 답하세요. 설명 없이 문서명만.

예시: ["2025학년도_교육과정_전자공학과", "교육과정_가이드라인"]

### 선택된 문서 (JSON 배열):
"""
        
        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config()
            )
            
            result_text = (response.text or "").strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            selected_names = json.loads(result_text)
            
            if not isinstance(selected_names, list):
                selected_names = [selected_names]
            
            # 이름을 파일명으로 매핑
            selected_files = []
            for name in selected_names:
                clean_name = name.strip()
                for filename in available_indices:
                    doc_name = filename.replace("_index.json", "")
                    if doc_name == clean_name or filename == clean_name:
                        if filename not in selected_files:
                            selected_files.append(filename)
                        break
                    if clean_name in doc_name or clean_name in filename:
                        if filename not in selected_files:
                            selected_files.append(filename)
                        break
            
            if selected_files:
                print(f" Router selected {len(selected_files)}/{len(available_indices)} documents")
                return selected_files
            else:
                print("Router couldn't match documents, using all")
                return available_indices
        
        except Exception as e:
            print(f"Router failed: {e}, using all documents")
            return available_indices
