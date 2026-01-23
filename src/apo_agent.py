import asyncio
import json
import os
from typing import Dict, TypedDict, List
import agentlightning as agl
from src.core.reasoner import RegulatoryReasoner
from src.config import Config


class QueryTask(TypedDict):
    id: str
    question: str
    expected_answer: str
    index_filename: str


SYSTEM_PROMPT_TEXT = """당신은 규제 준수 컨설턴트입니다.
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
{comparison_section}

### 답변 구조:
1. 직접 답변 (페이지 참조 포함)
{comparison_instruction}
3. 📚 참조 페이지 요약"""

COMPARISON_SECTION_TEXT = """
### 📊 다중 문서 비교 분석 (필수):
여러 문서가 제공되었으므로, 반드시 다음 형식으로 비교 분석을 포함하세요:

**1. 공통점 (Commonalities)**
- 모든 문서에서 일치하는 내용
- 예: "모든 교육과정에서 졸업 학점은 130학점 이상 [문서A, p.5], [문서B, p.3]"

**2. 차이점 (Differences)**
표 형식으로 명확히 구분:
| 항목 | 문서1 | 문서2 |
|------|------|------|
| 예: 필수학점 | 18학점 [p.5] | 21학점 [p.4] |
| 예: 선택과목 | 10개 [p.7] | 15개 [p.6] |

**3. 규제 우선순위**
- 충돌하는 규정이 있다면, 어떤 문서가 상위 규정인지 명시
- 예: "ISO가 상위 표준이므로 우선 적용 [ISO, p.10]"
"""


@agl.rollout
async def regulatory_agent(task: QueryTask, llm: agl.LLM) -> str:
    """Agent Lightning rollout 함수"""
    
    index_filenames = task["index_filename"].split(",")
    
    reasoner = RegulatoryReasoner(index_filenames)
    
    combined_context = []
    for idx, tree in enumerate(reasoner.index_trees):
        doc_name = index_filenames[idx].replace("_index.json", "")
        combined_context.append({"document": doc_name, "content": tree})
    
    context_str = json.dumps(combined_context, ensure_ascii=False)
    
    is_multi_doc = len(index_filenames) > 1
    
    comparison_section = (
        COMPARISON_SECTION_TEXT if is_multi_doc else ""
    )
    comparison_instruction = (
        "2. 문서 비교 분석 (공통점/차이점 표)" if is_multi_doc else ""
    )
    
    system_prompt = SYSTEM_PROMPT_TEXT.format(
        comparison_section=comparison_section,
        comparison_instruction=comparison_instruction,
    )
    
    full_prompt = f"""{system_prompt}

### 컨텍스트 (다중 문서 인덱스):
{context_str}

### 질문:
{task["question"]}

### 답변 (위 규칙을 철저히 따라 작성):
"""
    
    response = await llm.generate(prompt=full_prompt, temperature=0.0)
    
    agl.emit_reward(
        value=calculate_reward(response, task["expected_answer"]),
        explanation=f"답변 품질: {response[:100]}..."
    )
    
    return response


def calculate_reward(response: str, expected_answer: str) -> float:
    """간단한 reward 계산"""
    if not response:
        return 0.0
    
    if expected_answer.lower() in response.lower():
        return 1.0
    
    has_citation = "[" in response and "p." in response
    has_summary = "📚" in response
    
    reward = 0.0
    if has_citation:
        reward += 0.5
    if has_summary:
        reward += 0.3
    
    return min(reward, 1.0)


async def debug():
    """디버그 실행"""
    print("🔍 Regulatory Agent Debug 시작...")
    
    eval_path = "data/eval_dataset.jsonl"
    if not os.path.exists(eval_path):
        print(f"❌ {eval_path} 파일을 찾을 수 없습니다.")
        return
    
    tasks = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    
    print(f"✅ {len(tasks)}개 태스크 로드됨")
    
    llm = agl.LLM(
        endpoint=os.getenv("OPENAI_BASE_URL", "http://localhost:8000"),
        model="gemini-2.5-flash",
        sampling_parameters={"temperature": 0.0},
    )
    
    for task in tasks[:2]:
        print(f"\n📝 질문: {task['question']}")
        try:
            response = await regulatory_agent(task, llm)
            print(f"✅ 답변: {response[:200]}...")
        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    asyncio.run(debug())
