import json
import os
from typing import Any
from src.config import Config

class RegulatoryReasoner:
    def __init__(self, index_filename: str):
        path = os.path.join(Config.INDEX_DIR, index_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.index_tree = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in index file: {e}")
        except IOError as e:
            raise IOError(f"Failed to read index file: {e}")

        self.index_filename = index_filename

    def query(self, user_question: str) -> str:
        if not user_question or not user_question.strip():
            raise ValueError("user_question cannot be empty")
        
        context_str = json.dumps(self.index_tree, ensure_ascii=False)

        prompt = f"""
        당신은 규제 준수 컨설턴트입니다.
        제공된 규제 인덱스 JSON을 사용하여 사용자의 질문에 정확하게 답변하세요.
        
        ### 규칙:
        1. 제공된 인덱스를 기반으로만 엄격하게 답변하세요.
        2. **추적 가능성**: 모든 주장에 대해 반드시 섹션 ID와 페이지 번호를 인용해야 합니다. (예: [Sec 3.1, Pg 12])
        3. 답변이 여러 섹션과 관련된 경우, 논리적 연결을 설명하세요.

        ### 컨텍스트 (인덱스):
        {context_str}

        ### 질문:
        {user_question}
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