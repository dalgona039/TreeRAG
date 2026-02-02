import json
import os
from typing import Any, List, Dict
from src.config import Config

class TreeRAGReasoner:
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

    def query(self, user_question: str, enable_comparison: bool = True) -> str:
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

        prompt = f"""
당신은 전문 문서 분석 AI 어시스턴트입니다.
제공된 문서의 인덱스를 사용하여 사용자의 질문에 정확하게 답변하세요.

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