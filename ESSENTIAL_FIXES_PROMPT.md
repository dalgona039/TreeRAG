# PageTree-RAG — 필수 보강 실행 프롬프트 (Claude Code)

> 목적: TIST 리뷰 대응의 *필수 3항목*을 리포에서 실행한다.
> (1) **Language-drift 수정**(버그), (2) **Hallucination detector에 semantic 신호 추가**,
> (3) **Self-generated QA 사람 검증**. 각 항목은 산출물을 남기고, 그 수치로 논문을 갱신한다.
> 백엔드는 로컬 Ollama `llama3.1:8b`(seed 42), 임베딩은 dense retrieval과 동일 모델 사용.

## 절대 규칙
1. 수치 날조 금지 — 표/본문 수치는 실행 산출 JSON에서만 인용. 실패하면 보고.
2. 각 단계 후 산출물 확인 → 이전 대비 **델타 보고**(예: 한국어 답변율 28%→N%).
3. docx 편집이 필요하면 Word에서 파일을 **닫고** 진행. 매 편집 전 백업.
4. seed=42 고정. 재현 가능하게 스크립트/프롬프트를 커밋.

---

## Task 1 — Language drift 수정 (가장 높은 ROI, 근본원인 = 버그)

**증상(에러분석 실측):** 로컬 8B가 영어 HotpotQA 질문에 **한국어로 28/100** 답변 →
ROUGE≈0, LLM-Judge 소폭 하락. 검색이 아니라 **생성 프롬프트** 문제.

**근본원인:** `benchmarks/run_real_evaluation.py`에 답변 언어가
`"답변은 한국어로 작성하세요."`로 하드코딩됨(약 L162). TreeRAG 계열 생성은
reasoner/traverser·도메인 템플릿을 거치므로 거기에도 언어 지시가 있을 수 있음.

**할 일:**
```bash
# 1) 생성 경로의 언어 지시를 모두 찾는다
grep -rn "한국어\|답변은\|in Korean\|language" benchmarks/run_real_evaluation.py src/core/reasoner.py src/core/*prompt* src/core/domain_benchmark.py
```
- 답변 언어를 **질문 언어에 맞추도록** 변경한다. 최소한 영어 벤치마크(HotpotQA, GovReport)는
  "Answer in the same language as the question (English)."로. 데이터셋 메타에 `lang` 필드가
  있으면 그걸 사용, 없으면 질문에서 간단 감지(예: HotpotQA/GovReport=영어).
- 하드코딩된 한국어 지시는 **조건부**로 바꾸고, TreeRAG·baseline **양쪽 생성 경로 모두** 반영.

**재실행(영향 받은 평가):**
```bash
python benchmarks/run_exp2_multihop.py --n 100 --seed 42        # HotpotQA (fair)
python benchmarks/run_real_evaluation.py --dataset benchmarks/datasets/govreport.json \
  --systems all --limit 40 --seed 42 --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output data/benchmark_reports/online_local_llama_govreport_v2.json
python benchmarks/run_real_evaluation.py --dataset benchmarks/datasets/full_benchmark.json \
  --systems all --limit 100 --seed 42 --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output data/benchmark_reports/online_local_llama_general_v4_n100.json
```
**보고:** 각 산출물에서 **한국어 답변율**(정규식 `[가-힣]`)과 시스템별 ROUGE-L·LLM-Judge를
이전(v3/govreport/exp2) 대비 표로. 드리프트가 줄고 ROUGE가 오르면 그 방향으로 논문 갱신.
⚠️ 언어 통일이 LLM-Judge 순위를 바꾸면 **사실대로** 보고(멀티홉 서사 재점검 필요).

---

## Task 2 — Hallucination detector에 semantic(embedding) 신호 추가

**현재:** 5개 신호(citation presence, weighted word overlap, bigram, trigram, char n-gram)
평균. 리뷰어: "그냥 lexical 아니냐". 위치를 먼저 찾는다:
```bash
grep -rn "bigram\|trigram\|char.*overlap\|citation_presence\|confidence\|five.signal" \
  src/core src/api | grep -vi test
```
**할 일:**
- **6번째 신호 = semantic similarity**: 생성 문장 임베딩과 검색된 노드(근거) 임베딩 간
  **최대 코사인 유사도**. 임베딩은 dense retrieval과 동일 모델 재사용(`src/core/dense_retrieval_baseline.py` 참고)
  → 새 모델 의존성 없이 real-time 유지.
- confidence를 6신호로 확장(우선 동일 가중 평균; 여유 있으면 신호별 가중치/로지스틱은 후속).
- 문서화: "속도-정확도 균형은 유지하되 표면 매칭 한계를 semantic 신호로 보완."

**평가/검증:**
- 라벨(지원/비지원 문장)이 있으면 6신호 vs 5신호의 **AUROC/상관**을 비교해 개선 보고.
- 라벨이 없으면, error analysis에서 나온 **오답 사례(judge≤0.4)에서 detector confidence가
  더 잘 낮아지는지**를 표본으로 정량화(예: 오답에서 평균 confidence 하락폭).
```bash
python scripts/eval_detector.py --signals 5   # 없으면 작성
python scripts/eval_detector.py --signals 6
```
**보고:** 5신호 대비 6신호의 개선 수치 → 논문 §Hallucination Detection·Limitations 갱신.

---

## Task 3 — Self-generated QA 사람 검증

**목적:** 리뷰어 "LLM이 만든 QA는 bias 있다" 대응. 표본을 사람이 검증하고 유효율 보고.

**할 일:**
```bash
# 1) full_benchmark(+medical)에서 50문항 무작위 표본 → 주석 CSV 생성
python - <<'PY'
import json, random, csv
random.seed(42)
qs=json.load(open("benchmarks/datasets/full_benchmark.json"))["questions"]
sample=random.sample(qs, 50)
with open("benchmarks/human_eval/qa_verification_tasks.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["qid","question","expected_answer_hint",
        "answerable(1/0)","answer_correct(1/0)","grounded_in_source(1/0)","note"])
    for q in sample: w.writerow([q.get("id",""),q.get("question",""),q.get("expected_answer_hint",""),"","","",""])
print("wrote benchmarks/human_eval/qa_verification_tasks.csv (50 rows)")
PY
```
- **사람 2~3명**이 각 문항에 answerable/correct/grounded(1/0) 표시(자동화 불가).
- 채워진 CSV로 **유효율 + annotator 일치도(Cohen's κ)** 계산:
```bash
python benchmarks/human_eval/compute_agreement.py --input benchmarks/human_eval/qa_verification_tasks.csv
# (기존 compute_agreement 인터페이스에 맞게 컬럼 지정)
```
**보고:** "표본 50문항 중 X% answerable, Y% correct, Z% grounded, κ=…" → 논문 §Experimental
Setup(Dataset)에 1~2문장 + 소표로 추가.

**(보너스) 답변 human study**: `benchmarks/human_eval/generate_annotation_tasks.py`로
시스템 답변 표본을 뽑아 5명 평가 → `compute_agreement.py`로 LLM-Judge와의 상관 보고.

---

## 실행 순서 & 마무리
1. **Task 1(언어)** 먼저 — 재실행이 다른 수치에 영향을 주므로. → 새 리포트로 논문 표(8·10·12)와
   error-analysis 수치 갱신 필요(제게 산출물 주시면 반영).
2. Task 2(detector) → §Hallucination·Limitations 갱신.
3. Task 3(QA 검증) → §Dataset 갱신.
- 각 Task 종료 시: 산출 JSON/CSV 경로 + 이전 대비 델타를 요약해 남길 것.
- 완료 후 그 수치들을 알려주면, 논문(docx) 반영은 제가 처리합니다.
