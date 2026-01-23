# APO (Automatic Prompt Optimization) 가이드

## 개요

이 프로젝트는 Microsoft의 **Agent Lightning** 프레임워크를 사용하여 규제 문서 질의응답 Agent의 프롬프트를 자동으로 최적화합니다.

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 디버그 실행 (APO 없이 테스트)

```bash
python src/apo_agent.py
```

평가 데이터셋의 처음 2개 질문으로 Agent를 테스트합니다.

### 2. APO 학습 실행

```bash
python train_apo.py
```

**학습 과정:**
1. `data/eval_dataset.jsonl`에서 평가 데이터 로드
2. 초기 프롬프트로 성능 측정
3. APO 알고리즘으로 프롬프트 개선 (5회 반복)
4. 최적화된 프롬프트를 `data/optimized_prompts/` 저장

**예상 소요 시간:** 5-10분 (데이터셋 크기에 따라)

### 3. 최적화된 프롬프트 확인

```bash
cat data/optimized_prompts/system_prompt.txt
cat data/optimized_prompts/comparison_section.txt
```

### 4. 최적화된 프롬프트 적용

`src/core/reasoner.py`를 수정하여 최적화된 프롬프트를 사용하세요:

```python
# data/optimized_prompts/system_prompt.txt의 내용을 붙여넣기
```

## 평가 데이터셋 형식

`data/eval_dataset.jsonl`:

```jsonl
{"id": "q001", "question": "질문", "expected_answer": "예상 답변", "index_filename": "파일명.json"}
```

## APO 설정

`train_apo.py`에서 하이퍼파라미터 조정:

```python
apo_config = {
    "max_iterations": 5,      # 최적화 반복 횟수
    "beam_size": 3,           # 탐색할 프롬프트 후보 수
    "batch_size": 2,          # 배치 크기
    "learning_rate": 0.1,     # 학습률
}
```

## 작동 원리

1. **초기 프롬프트**: `SYSTEM_PROMPT`, `COMPARISON_SECTION` 템플릿 정의
2. **Rollout**: 각 질문에 대해 Agent 실행 → 답변 생성
3. **Reward 계산**: 
   - 예상 답변 포함 여부
   - 인용(citation) 존재 여부
   - 출처 요약 존재 여부
4. **프롬프트 개선**: APO가 reward를 최대화하는 프롬프트 탐색
5. **반복**: 더 나은 프롬프트 발견 시 업데이트

## 문제 해결

### API 키 오류
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### 데이터셋 없음
`data/eval_dataset.jsonl` 파일이 있는지 확인하세요.

### 메모리 부족
`batch_size`를 1로 줄이거나 `max_iterations`를 줄이세요.

## 참고 자료

- [Agent Lightning 공식 문서](https://microsoft.github.io/agent-lightning/)
- [APO 알고리즘 설명](https://microsoft.github.io/agent-lightning/algorithm-zoo/apo/)
- [예제 코드](https://github.com/microsoft/agent-lightning/tree/main/examples/apo)
