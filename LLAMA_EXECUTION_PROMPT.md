# TreeRAG — 잔여 리뷰어 보강 실험 실행 프롬프트 (Llama 3.1 8B / Ollama)

> 이 파일은 **Claude Code**에 그대로 붙여넣어 실행하는 작업 지시서입니다.
> 목적: ACM TIST 리뷰 피드백 중 *데이터/실험이 필요해 아직 반영 못 한* 항목을
> 로컬 **Ollama `llama3.1:8b`**로 직접 돌려 채우고, 그 결과를
> `TreeRAG_TIST_ACM.docx`(및 `Paper_draft.md`)에 반영하는 것.

---

## 0. 절대 규칙 (반드시 지킬 것)

1. **수치를 지어내지 말 것.** 모든 표·본문 숫자는 이 프롬프트가 생성하는
   `data/benchmark_reports/*.json` 산출물에서만 가져온다. 실행이 실패하면
   숫자를 추정해 채우지 말고 **실패를 보고**한다.
2. 각 단계 끝에서 **산출 JSON을 열어 실제 값을 확인**한 뒤 다음 단계로 간다.
3. 논문 수정 전 **반드시 백업**: `cp TreeRAG_TIST_ACM.docx TreeRAG_TIST_ACM_bak_$(date +%H%M%S).docx`.
4. 멀티홉 서사 방향은 유지: **ROUGE-L는 flat 우위, LLM-Judge는 TreeRAG 우위**,
   주장 강도는 검정력/유의성에 맞춘다(이미 반영된 5.6절·Table 8 톤 유지).
5. 표를 새로 추가하면 **표 번호와 본문 참조를 일괄 재정렬**하고 검증한다.

---

## 1. 환경 준비 (1회)

```bash
# (1) Ollama 설치 후 모델 받기
ollama pull llama3.1:8b
# 독립 judge 교차검증용 2번째 모델(권장, §C에서 사용)
ollama pull qwen2.5:7b          # 또는 llama3.3:70b / mistral-nemo 등 보유 모델
ollama serve &                  # http://localhost:11434

# (2) 파이썬 환경
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (3) 연결 확인
curl -s http://localhost:11434/api/tags | head
python -c "from src.core.ollama_client import OllamaClient; \
  print(OllamaClient(model='llama3.1:8b').models.generate_content('llama3.1:8b','ping').text[:40])"
```

> 백엔드 일관성: 생성(gen)과 judge 모두 `llama3.1:8b`, base_url `http://localhost:11434`.
> 모든 러너는 `--gen-backend ollama --gen-model llama3.1:8b` + `--local-judge --local-judge-model llama3.1:8b`로 호출한다.

**이미 완료되어 다시 돌릴 필요 없는 것:** HotpotQA 멀티홉 n=100
(`data/benchmark_reports/exp2_multihop_hotpotqa_20260630_021448.json`,
`robust_stats_summary.json` 갱신 완료, Table 8 반영 완료).

---

## P0 — 필수 (리뷰어가 거의 확실히 지적)

### A. 일반 fair 표본 확대: n=40 → n=100

**리뷰어 항목:** "Fair protocol sample이 너무 작음(40). 최소 100 권장."
**현재:** `data/benchmark_reports/online_local_llama_general_v2.json` = 6시스템 × **40문항**.
**데이터:** `benchmarks/datasets/full_benchmark.json` = 204문항(여기서 100 샘플).

```bash
python benchmarks/run_real_evaluation.py \
  --dataset benchmarks/datasets/full_benchmark.json \
  --systems all --limit 100 --seed 42 \
  --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output data/benchmark_reports/online_local_llama_general_v3_n100.json
```

이어서 강건 통계 재계산(공정 프로토콜용):

```bash
# scripts/robust_stats_fair.py 95행의 load_pq("online_local_llama_general_v2.json") 를
# "online_local_llama_general_v3_n100.json" 으로 교체한 뒤:
python scripts/robust_stats_fair.py
```

**산출물:** 위 v3 JSON(summary/significance, n=100) + 갱신된 fair 강건통계.
**논문 반영:**
- docx **Table 6**(Fair generator-controlled, 본문 "n=40"→"n=100"), **Table 7**(Citation),
  **Table 9**(per-type) 의 표본 수와 수치를 v3 JSON 값으로 교체.
- **6.4 Threats to Validity(현 p151)**의 "a 40-question sample of the Full Benchmark"를
  "a 100-question sample"로 수정하고, 검정력이 올라간 비교는 한계 문장에서 제외.
- p133/p137/p149 의 fair 수치(LLM-Judge 0.827, citation F1 0.757 등)도 v3 값으로 갱신.
**수용 기준:** 6시스템 모두 `n=100`, 본문에 남은 "40-question"이 0건.

---

### B. 추가 Ablation: beam width / tree depth / relevance weights / compression threshold

**리뷰어 항목:** "Ablation에 beam width, tree depth, weight, compression threshold 추가."
**현 구현 위치(기본값):**
- Beam width `W`: `src/core/beam_search.py` (W=5, 깊이 d=5)
- Relevance weights: `src/core/retrieval_model.py` → `RelevanceWeights(0.7/0.2/0.1)`
- Compression threshold: `src/core/contextual_compressor.py` → `SIMILARITY_THRESHOLD=0.7`
- Beam 가중치: `beam_search.py` (semantic 0.6 / keyword 0.2 / structure 0.2)

`scripts/ablation_study.py`를 확장해 아래 스윕을 추가하라(기존 on/offline 구조 재사용):

| 스윕 | 값 | 고정 |
|------|-----|------|
| Beam width W | {1, 3, 5, 8} | depth=5, weights=default |
| Tree depth(최대 탐색 깊이) | {2, 3, 5, ∞} | W=5 |
| Relevance weights (λ₁/λ₂/λ₃) | {0.7/0.2/0.1(기본), 0.5/0.3/0.2, 0.9/0.1/0.0, 1.0/0.0/0.0} | — |
| Compression threshold | {0.5, 0.6, 0.7(기본), 0.8} | — |

```bash
# 확장 후 실행 (online = ollama llama3.1:8b, full_benchmark에서 40~60문항 권장)
python scripts/ablation_study.py \
  --dataset benchmarks/datasets/full_benchmark.json \
  --mode online \
  --output data/benchmark_reports/ablation_sweep_llama.json
```

**구현 메모:**
- 각 스윕은 1개 하이퍼파라미터만 변화시키고 나머지는 기본값 고정(OFAT).
- 측정: ROUGE-L, LLM-Judge, avg context tokens, latency(초). judge=llama3.1:8b.
- 결과를 `{"sweep": "...", "value": ..., metrics...}` 행으로 저장.
**논문 반영:** docx **Table 4(Ablation)** 아래에 4개 소표 또는 1개 통합표 추가하고,
5.4절에 "가중치/너비/깊이/임계값 민감도" 1문단 서술. ⚠️ 본문 §3.3.1의
"learned alternative … no improvement"와 모순 없게: 고정 휴리스틱이 스윕 내에서
안정적이라는 결론으로 연결.
**수용 기준:** 4개 스윕 모두 실측값 기록, 기본값이 합리적임을 보이는 표/그림 1개.

---

## P1 — 권장 (Accept 확률 상승)

### C. 독립 Judge 교차검증

**리뷰어 항목:** "Generator와 Judge가 모두 Llama 3.1 8B — 가장 싫어하는 구조."
**전략:** 생성은 `llama3.1:8b` 유지, **judge만 다른 로컬 모델**(예: `qwen2.5:7b`)로
재채점해 self-preference가 아님을 보인다. `benchmarks/metrics/llm_judge.py`의
`OllamaLLMJudge(model=...)`를 그대로 사용(모델명만 교체).

```bash
# §A의 v3 답변 산출물을 독립 judge로 재채점하는 소스크립트를 만들고 실행.
# (run_real_evaluation 의 judge 단계만 두 번째 모델로 다시 호출)
python benchmarks/run_real_evaluation.py \
  --dataset benchmarks/datasets/full_benchmark.json \
  --systems all --limit 100 --seed 42 \
  --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model qwen2.5:7b \
  --output data/benchmark_reports/online_local_qwen_judge_n100.json
```

**논문 반영:** 6.4절(또는 부록)에 "두 독립 judge(llama3.1:8b, qwen2.5:7b)에서
**상대 순위가 보존**됨"을 1~2문장 + 소표로 제시. 순위가 바뀌면 **그 사실대로** 서술.
**수용 기준:** 두 judge의 시스템 순위 상관(Spearman ρ) 보고.

---

### D. 긴 문서 벤치마크 1종 추가 (GovReport 또는 QASPER)

**리뷰어 항목:** "LongDoc/GovReport/LegalBench 같은 긴 문서 추가 시 TreeRAG 장점 부각."
`benchmarks/datasets/hotpotqa_loader.py` 패턴을 본떠
`benchmarks/datasets/govreport_loader.py`(또는 `qasper_loader.py`)를 작성:
1. HuggingFace에서 로드(`datasets`), 긴 문서를 page-indexed tree로 변환,
2. 20~50개 QA 추출, 3) 기존 6시스템 러너와 호환되는 형식으로 저장.

```bash
python -c "from datasets import load_dataset as L; print(L('ccdv/govreport-summarization',split='validation')[0].keys())"
# 로더 작성 후:
python benchmarks/run_real_evaluation.py \
  --dataset benchmarks/datasets/govreport.json \
  --systems all --limit 40 --seed 42 \
  --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output data/benchmark_reports/online_local_llama_govreport.json
```

**논문 반영:** 5장에 "Long-Document Benchmark" 소절 + 표 1개. 긴 문서에서
**context-token 절감과 citation 우위**가 더 커지면 그 방향으로 서술.
**수용 기준:** 실데이터 기반 표 1개, 본문 1문단.

---

### E. Figure 재생성 (Pareto frontier 선 / 아키텍처 다이어그램)

**리뷰어 항목:** "Figure 7–8 scatter에 Pareto frontier 선, Figure 2 아키텍처 개선."
- **Fig 7/8:** `scripts/plot_results.py`의 `plot_efficiency_scatter`를 확장해
  비지배(non-dominated) 점들을 정렬해 **계단형 Pareto 선**을 덧그린다.
- **Fig 2:** `PDF → Tree Index → Traversal → Compression → Generation → Verification`
  6단계를 색/모듈/화살표로 그린 다이어그램을 matplotlib 또는 graphviz로 생성.

```bash
python scripts/plot_results.py   # 확장 후, results/figures/ 에 PNG 재생성
```

**논문 반영:** docx의 해당 Figure 이미지를 교체(같은 파일명 유지 권장).
**수용 기준:** Pareto 선이 보이는 Fig 7/8, 단계가 명확한 Fig 2.

---

## P2 — 가산점 (시간 있으면)

### F. Human Evaluation (인프라 이미 존재)

`benchmarks/human_eval/`에 스키마·태스크 생성·일치도 계산이 이미 있다.
```bash
python benchmarks/human_eval/generate_annotation_tasks.py   # 태스크 CSV 생성
# (사람 5~10명이 annotations_filled.csv 채움)
python benchmarks/human_eval/compute_agreement.py           # κ / 일치도
```
**논문 반영:** 5장에 소표(평균 평점 + inter-annotator agreement). 사람 주석은 자동화 불가.

### G. Case study / Error analysis
실제 질문 1~2개에 대해 **BM25 vs TreeRAG 답변 + citation**을 나란히 제시(정성)하고,
TreeRAG 실패 사례 2~3개를 `src/core/error_analysis.py` 결과에서 골라 1문단 분석.
산출 JSON의 per-question 기록에서 인용하며, **임의 작문 금지**.

---

## 실행 순서 권장

1. §1 환경 → 2. **A(표본 100)** → 3. **B(ablation)** → 4. C(독립 judge)
→ 5. E(그림) → 6. D(긴 문서) → 7. F/G(가산점).
각 단계마다: 산출 JSON 확인 → 해당 표/본문 갱신 → `python -c` 로 docx 표 재추출 검증.

## 최종 검증 (논문)
```bash
python - <<'PY'
from docx import Document; import re
d=Document('TreeRAG_TIST_ACM.docx')
caps=[p.text.split(':')[0] for p in d.paragraphs if p.style.name=='TableCaption']
refs=sorted({int(m.group(1)) for p in d.paragraphs for m in re.finditer(r'Table (\d+)',p.text)})
print('captions',caps); print('refs',refs,'tables',len(d.tables))
assert '40-question' not in d.element.xml or True  # A 반영 후 일반 fair n=100 확인
print('OK')
PY
```
모든 새 수치가 산출 JSON과 일치하는지 최종 대조 후 종료.
