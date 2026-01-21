import json
import os
from typing import Any, List, Dict
from src.config import Config

class RegulatoryReasoner:
    def __init__(self, index_filenames: List[str]):
        self.index_trees: List[Dict[str, Any]] = []
        self.index_filenames = index_filenames
        
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

    def query(self, user_question: str) -> str:
        if not user_question or not user_question.strip():
            raise ValueError("user_question cannot be empty")
        
        combined_context = []
        for idx, tree in enumerate(self.index_trees):
            doc_name = self.index_filenames[idx].replace("_index.json", "")
            combined_context.append({
                "document": doc_name,
                "content": tree
            })
        
        context_str = json.dumps(combined_context, ensure_ascii=False)

        prompt = f"""
        당신은 규제 준수 컨설턴트입니다.
        제공된 여러 규제 문서의 인덱스를 사용하여 사용자의 질문에 정확하게 답변하세요.
        
        ### 중요 규칙:
        1. **반드시 인덱스 데이터만 사용**: 제공된 인덱스에 없는 정보는 절대 추측하거나 생성하지 마세요.
        2. **페이지 번호 필수 표기**: 모든 문장마다 반드시 출처 페이지를 명시하세요.
           - 형식: [문서명, p.페이지번호] 또는 [문서명, p.시작-끝]
           - 예시: "교육과정은 4학기로 구성됩니다 [전자공학과_교육과정, p.5]"
        3. **여러 페이지 참조**: 정보가 여러 페이지에 걸쳐 있으면 모두 표기하세요.
           - 예시: [문서A, p.3-5, p.12]
        4. **문서 구조 활용**: 인덱스의 page_ref 필드를 정확히 사용하세요.
        5. **답변 끝에 출처 요약**: 답변 마지막에 참조한 모든 페이지를 나열하세요.
           - 형식: "📚 **참조 페이지**: [문서명, p.3], [문서명, p.7-9]"

        ### 답변 템플릿:
        [본문 내용 + 인라인 페이지 참조]
        
        📚 **참조 페이지**: [문서1, p.X], [문서2, p.Y-Z]

        ### 컨텍스트 (다중 문서 인덱스):
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
            return response.text
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise