# Citation Availability 버그 수정 프롬프트

> 새 대화에서 TreeRAG 폴더 첨부 후 아래 내용을 붙여넣으세요.

---

```
TreeRAG 프로젝트의 ACM 논문 평가 결과에서 Citation Availability가 모든 시스템에서 0.000으로 나오는 버그를 수정해주세요.

## 반드시 먼저 읽을 파일들

1. src/core/reasoner.py — 답변 생성 프롬프트 (페이지 참조 형식 정의)
2. benchmarks/run_real_evaluation.py — 평가 실행 스크립트
3. scripts/generate_paper_tables.py — 테이블 생성 (citation_rate 함수 위치: line ~569)
4. benchmarks/analysis/raptor_vs_treerag.py — citation_rate 함수 (_CITATION_RE regex)
5. data/benchmark_reports/evaluation_latest.json — 실제 평가 결과 (answer 필드 형식 확인)

## 현재 문제 상황

`generate_paper_tables.py`와 `raptor_vs_treerag.py`의 citation 감지 regex:
```python
_CITATION_RE = re.compile(r"\[[^\]]*p\.\s*\d+", re.IGNORECASE)
```
이 패턴은 `[문서명, p.5]` 형식만 찾습니다.

`reasoner.py` 프롬프트는 `[문서명, p.페이지]` 형식으로 citation을 요청하지만,
실제 평가 결과(`evaluation_latest.json`)의 `answer` 필드를 확인하면
LLM이 실제로 어떤 형식으로 citation을 출력하는지 확인이 필요합니다.

## 수행할 작업

### STEP 1: 실제 답변 형식 확인
`data/benchmark_reports/evaluation_latest.json`에서 treerag_beam 또는 treerag_dfs의
answer 필드 3~5개를 직접 읽어서 citation이 포함됐는지, 어떤 형식인지 파악하세요.

### STEP 2: 원인에 따라 아래 두 가지 중 하나(또는 둘 다) 수정

**Case A: LLM이 citation을 출력하지 않는 경우**
`src/core/reasoner.py`의 프롬프트를 수정하여 citation 출력을 더 강제하세요.
- `reasoner.py`를 읽지 않고 수정하지 마세요.
- 프롬프트 수정 후 캐시 무효화를 위해 `PROMPT_CACHE_VERSION`을 갱신하세요.

**Case B: LLM이 다른 형식으로 citation을 출력하는 경우**
`scripts/generate_paper_tables.py`의 `_citation_rate` 함수와
`benchmarks/analysis/raptor_vs_treerag.py`의 `_CITATION_RE` regex를
실제 출력 형식에 맞게 수정하세요.

### STEP 3: FlatRAG LLM-Judge 이상값 확인
`data/benchmark_reports/evaluation_latest.json`에서 FlatRAG의 llm_judge 점수가
0.81로 TreeRAG(0.70)보다 높은 이유를 파악하세요.
- LLM-Judge 프롬프트가 정확도보다 유창성을 평가하고 있는지 확인
- `benchmarks/run_real_evaluation.py`에서 llm_judge 프롬프트 찾기
- 문제가 있으면 수정하고, 단순히 FlatRAG 답변이 유창한 것이라면
  논문 Discussion에 넣을 설명 문장 1~2줄 제안만 해주세요 (코드 수정 불필요)

### STEP 4: 수정 후 재실행

수정 내용에 따라 필요한 명령만 실행하세요:

**Case A (reasoner 수정한 경우):**
```bash
# 캐시 초기화 후 전체 재평가
source .venv/bin/activate
python benchmarks/run_real_evaluation.py --systems treerag_dfs,treerag_beam --use-llm-judge --output data/benchmark_reports/evaluation_treerag_only.json
```
그 다음 기존 evaluation_latest.json의 bm25/dense/flatrag/raptor 결과와 병합하거나,
전체 재실행:
```bash
python benchmarks/run_real_evaluation.py --systems all --use-llm-judge
python benchmarks/run_real_evaluation.py --dataset hotpotqa --systems bm25,flatrag,raptor,treerag_beam --output data/benchmark_reports/hotpotqa_results.json
python benchmarks/run_real_evaluation.py --dataset medical --domain medical --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam --output data/benchmark_reports/medical_results.json
```

**Case B (regex만 수정한 경우):**
```bash
source .venv/bin/activate
python scripts/generate_acm_outputs.py
python benchmarks/analysis/raptor_vs_treerag.py
```

### STEP 5: 테이블 재생성 및 결과 확인
```bash
python scripts/generate_acm_outputs.py
cat data/benchmark_reports/paper_tables_acm.tex
```
Citation Avail. 컬럼에 TreeRAG > 0.000 이 나오면 성공입니다.

## 완료 기준

1. `paper_tables_acm.tex`의 Table 2에서 TreeRAG Citation Avail. > 0.000
2. LLM-Judge 이상값에 대한 설명 또는 수정 완료
3. 수정한 파일 목록과 변경 내용 요약 출력
4. `pytest -q --tb=short 2>&1 | tail -5` — 기존 통과 테스트 수 유지 확인

## 환경

- Python: .venv (프로젝트 루트의 `.venv/bin/activate`)
- API: `.env` 파일에 GOOGLE_API_KEY 등 설정됨
- 모델: gemini-2.5-flash (src/config.py의 Config.MODEL_NAME)
```
