# Claude Code 실행 프롬프트 — B그룹 (방법론 공백 해소)

사용법: TreeRAG 저장소 루트에서 Claude Code를 실행하고 아래 블록 전체를 붙여넣으세요.
B3·B5·B6는 코드 조사만으로 끝나므로 먼저 실행하고, B1·B7·B2는 실험이 필요하니 순서대로 진행하는 것을 권합니다.

---

## 붙여넣을 프롬프트

```
당신은 이 저장소(PageTree-RAG)의 논문 TreeRAG_TIST_ACM.docx를 ACM TIST 투고 수준으로
끌어올리는 작업을 돕고 있습니다. 논문 심사에서 반드시 문제가 될 방법론 공백 7건(B1~B7)을
해소하는 것이 목표입니다.

지켜야 할 원칙 — 매우 중요합니다:
- 수치는 반드시 코드를 실행하거나 로그/결과 파일에서 읽어 얻으십시오. 추정하거나
  그럴듯한 값을 만들어내지 마십시오.
- 알 수 없는 값은 "측정되지 않음"으로 보고하십시오. 빈칸을 채우려 하지 마십시오.
- 기존 실험 결과 파일(benchmarks/results/)을 덮어쓰지 말고 새 파일로 저장하십시오.
- 각 작업이 끝날 때마다 무엇을 측정했고 무엇을 측정하지 못했는지 명확히 보고하십시오.

작업은 아래 우선순위대로 진행합니다. 각 단계가 끝나면 결과를 보고하고 다음으로 넘어가십시오.

────────────────────────────────────────────────────────
[단계 0] 현황 파악 (코드 실행 없음)
────────────────────────────────────────────────────────
다음을 읽고 구조를 파악한 뒤 요약해 주십시오.

- src/core/tree_traversal.py, src/core/beam_search.py  (traversal, LLM 호출 지점)
- src/core/bm25_baseline.py, dense_retrieval_baseline.py, flat_rag_baseline.py,
  raptor_baseline.py                                    (baseline 구성)
- src/core/contextual_compressor.py                     (압축, 토큰 카운트)
- benchmarks/metrics/text_similarity.py, citation_metrics.py, llm_judge.py,
  efficiency_metrics.py                                 (지표 구현)
- benchmarks/compare_baselines.py, run_evaluation.py, run_real_evaluation.py,
  run_exp1_citation.py, run_exp2_multihop.py, run_exp3_efficiency.py
- benchmarks/results/ 안에 어떤 실행 결과가 남아 있는지 목록

보고 형식: 각 파일이 무엇을 하는지 1~2줄, 그리고 "논문의 Table N을 만드는 스크립트는
X이다"라는 대응표.

────────────────────────────────────────────────────────
[단계 1] B3 — baseline 하이퍼파라미터 추출 (코드 조사만)
────────────────────────────────────────────────────────
논문 §5.1의 Baselines 문단에는 재현에 필요한 값이 하나도 없습니다. 코드에서 실제 사용된
값을 찾아 다음 표를 채우십시오. 코드에 하드코딩된 기본값과 실험 스크립트가 넘긴 값이
다르면 둘 다 적고 실제 실행에 쓰인 쪽을 표시하십시오.

| 항목 | 값 | 출처 (파일:줄) |
|---|---|---|
| 청크 크기 (토큰 또는 문자) | | |
| 청크 overlap | | |
| top-k (BM25 / Dense / FlatRAG / RAPTOR 각각) | | |
| BM25 k1, b | | |
| Dense: Sentence-BERT 체크포인트명 | | |
| Dense: 유사도 함수, 정규화 여부 | | |
| FlatRAG: 60/25/15 가중치의 정의와 출처 | | |
| RAPTOR: 클러스터 수/방법, 요약 모델, 트리 높이 | | |
| PageTree-RAG: max_branches b, max_depth d, beam width W | | |

★ 청크 크기는 논문의 "10× fewer context tokens" 주장의 분모이므로 반드시 확정해야 합니다.
찾지 못하면 "코드에서 확인 불가"라고 명시하십시오.

산출물: docs/paper_revision/B3_baseline_hyperparams.md

────────────────────────────────────────────────────────
[단계 2] B5 — 지표 변형 확정 (코드 조사 + 소규모 재계산)
────────────────────────────────────────────────────────
논문은 ROUGE-L을 "longest-common-subsequence recall", BERTScore를 "contextual embedding
precision"이라고 서술합니다. 둘 다 통상 F 계열로 보고하는 지표이며, recall과 precision은
답변 길이에 정반대 방향으로 편향됩니다. 논문 전체의 논지가 ROUGE-L 순위 역전에 걸려
있으므로 이 부분이 흔들리면 안 됩니다.

benchmarks/metrics/text_similarity.py 등을 읽고 확정하십시오:
1. ROUGE-L: 실제로 recall인가 precision인가 F인가? 어떤 라이브러리(rouge-score,
   rouge, 자체 구현)인가? stemming/tokenizer 설정은?
2. BERTScore: precision/recall/F 중 무엇인가? 어떤 모델 체크포인트인가?
   baseline rescaling을 켰는가?
3. 위 답을 논문 서술과 대조해 일치/불일치를 표로 정리.

그 다음, 이미 저장된 fair protocol(n=100) 답변 파일이 있다면 재생성 없이 다음을
계산하십시오. 없으면 "저장된 답변 없음"이라 보고하고 넘어가십시오.
- 시스템별 평균 답변 길이(토큰/단어)와 표준편차
- ROUGE-L의 P/R/F 세 변형을 모두 계산한 표
- BERTScore의 P/R/F 세 변형을 모두 계산한 표

★ 목적: "ROUGE-L 역전이 지표 변형 선택의 산물이 아니다"를 보이는 것입니다. 만약 F로
바꾸면 순위가 달라진다면 그것 자체가 논문에 반드시 실려야 할 중요한 발견입니다.
결과가 논문에 불리해도 그대로 보고하십시오.

산출물: docs/paper_revision/B5_metric_variants.md (+ 계산 스크립트와 결과 CSV)

────────────────────────────────────────────────────────
[단계 3] B6 — 언어 불일치 교란변수 점검
────────────────────────────────────────────────────────
논문 Appendix A.2는 traversal 프롬프트가 릴리스 구현에서 한국어로 작성돼 있다고 밝힙니다.
그런데 HotpotQA와 GovReport는 영어 벤치마크입니다. §6.3은 생성기가 영어 질문에 한국어로
답하는 버그(100건 중 28건)를 보고합니다.

확인할 것:
1. src/core/tree_traversal.py와 beam_search.py의 노드 채점 프롬프트가 실제로 한국어인가?
   질의 언어에 따라 전환되는가, 아니면 고정인가?
2. 영어 벤치마크 실행 시에도 한국어 프롬프트가 쓰였는가? (실행 로그/설정으로 확인)
3. §6.3의 한국어 출력 버그는 traversal 프롬프트에서 왔는가, 생성 프롬프트에서 왔는가?
   커밋 히스토리나 수정 지점을 찾아 확인하십시오.
4. 이 언어 불일치는 PageTree-RAG 경로에만 존재합니다(baseline은 LLM 채점을 하지 않으므로).
   즉 PageTree-RAG에만 적용되는 교란변수입니다.

가능하면 소규모 대조 실험을 하십시오: HotpotQA 30문항에 대해 traversal 프롬프트를
영어로 번역한 버전과 기존 한국어 버전을 각각 실행하고 LLM-Judge/ROUGE-L을 비교.
차이가 유의하지 않다면 그 사실이 논문의 방어 근거가 됩니다.

산출물: docs/paper_revision/B6_prompt_language.md

────────────────────────────────────────────────────────
[단계 4] B1 — 파이프라인 전체 토큰 회계 ★가장 중요★
────────────────────────────────────────────────────────
논문은 generator에 들어간 컨텍스트 토큰만 셉니다. 그러나 PageTree-RAG는 질의마다 노드
관련성 채점을 위해 다수의 LLM 호출을 합니다. 그 토큰은 어디에도 집계되지 않습니다.
질의당 LLM 호출 수는 논문 전체에 한 번도 등장하지 않습니다.

제목과 초록에 "Token-Efficient"를 넣은 이상 리뷰어는 반드시 총 비용을 묻습니다.
지금은 회계의 절반만 계산하고 있으므로, 총 토큰으로는 BM25보다 훨씬 클 가능성이 높습니다.

할 일:
1. traversal 경로(DFS, Beam)와 압축, 생성, 판정에 이르는 모든 LLM 호출 지점에 계측을
   추가하십시오. 호출마다 다음을 기록: 단계 이름, 모델명, 입력 토큰, 출력 토큰, wall-clock.
   기존 동작을 바꾸지 말고 계측만 덧붙이십시오(환경변수로 on/off 가능하게).
2. fair protocol과 동일한 설정으로 Full Benchmark 100문항(seed 42)을 6개 시스템 전부에
   대해 재실행하고, 질의당 다음을 집계하십시오:

   | System | LLM calls/query | Prompt tok/query | Completion tok/query | Total tok/query | Generator ctx tok | Latency (s) |

   baseline들은 생성 1회뿐이므로 호출 수가 1일 것입니다. 그 대비가 핵심입니다.
3. 호스팅 API 단가를 하나 골라(예: 실제로 쓴 gemini-2.5-flash) 질의당 추정 비용을
   덧붙이십시오. 단가는 출처를 명시하십시오.
4. 결과를 논문의 새 표(Table 8 바로 뒤에 들어갈 "Table 8b: End-to-end token accounting")
   형태의 마크다운 표로 출력하십시오.

★ 결과가 "PageTree-RAG의 총 토큰이 baseline보다 많다"로 나와도 그대로 보고하십시오.
   그 경우 논문의 서술은 "생성 단계 컨텍스트를 검색 단계 비용과 교환했다"로 바뀌어야
   하며, 이는 정직한 프레이밍이자 오히려 방어 가능한 주장입니다.

산출물:
- docs/paper_revision/B1_token_accounting.md (표 + 해석)
- benchmarks/results/token_accounting_<날짜>.json
- 계측 코드 (기존 경로에 부작용 없이)

────────────────────────────────────────────────────────
[단계 5] B7 — 대규모 평가가 실제 시스템이 아닌 문제
────────────────────────────────────────────────────────
논문 §4.5는 "offline keyword-scoring path used for large-scale evaluation (Section 5)"가
LLM API 장애 시의 fallback이라고 밝힙니다. 그렇다면 Table 3(n=204), Table 4(medical n=42),
Table 5(ablation n=70), Table 7(n=204), Table 13의 General/Medical 행 전체가 LLM 노드
채점을 쓰지 않은 축소 버전의 결과입니다. 즉 논문의 핵심 기여인 LLM 기반 적응형 traversal이
이 결과들에는 작동하지 않았습니다.

할 일:
1. 코드에서 offline/keyword-only 경로와 full LLM 경로를 식별하고, 위 각 표가 어느 경로로
   생성됐는지 확정하십시오. 확정 결과를 표로 정리.
2. Medical Benchmark(n=42)를 fair generative protocol로 재실행하십시오. 규모가 작아
   비용이 감당 가능하고, medical entity recall 1.000은 논문에서 가장 인상적인 수치인데
   현재는 fallback 경로 결과이므로 우선순위가 높습니다.
   - 6개 시스템 전부, 동일 생성기(Llama 3.1 8B), seed 42
   - ROUGE-L, BERTScore, LLM-Judge, Medical Entity Recall, Citation Availability/F1,
     Context Tokens, Latency
3. 가능하면 Ablation(Table 5)도 fair protocol로 재실행. 비용이 크면 건너뛰고 이유를
   보고하십시오.
4. 재실행이 불가능한 표에 대해서는 캡션에 넣을 문구를 제안하십시오. 예:
   "keyword-only scoring path; LLM node judgment disabled (Section 4.5)".

산출물: docs/paper_revision/B7_protocol_audit.md + 재실행 결과 JSON

────────────────────────────────────────────────────────
[단계 6] B2 — Citation F1의 gold 순환성 해소
────────────────────────────────────────────────────────
Full Benchmark는 LLM이 source node에서 질문을 생성하고 answer hint가 그 노드에 verbatim으로
있는지 검증해 만들었습니다. 그렇다면 gold supporting section = 질문을 생성한 그 노드이고,
이는 PageTree-RAG의 검색 단위와 동일합니다. 청크 기반 시스템은 구조적으로 불리합니다.
헤드라인 결과(citation F1 = 0.757)가 이 설계 때문이라는 반박이 나올 수 있습니다.

할 일:
1. benchmarks/metrics/citation_metrics.py를 읽고 Citation Availability와 Citation F1의
   정확한 정의를 수식으로 적으십시오. 특히 다음을 확정:
   - Availability는 생성된 답변 텍스트를 보는가, 검색된 컨텍스트를 보는가?
   - F1의 예측 집합과 gold 집합은 각각 무엇인가?
   - 청크 기반 시스템의 출력이 gold section과 어떻게 매칭되는가?
   ★ 현재 논문 Table 9는 FlatRAG가 Availability 0.000인데 F1 0.570입니다. 이 조합이
     코드상 어떻게 가능한지 반드시 규명하고, 논문 캡션의 설명(1차 수정에서 추가함)이
     실제 구현과 맞는지 확인하십시오. 다르면 캡션을 실제 정의로 교체해야 합니다.
2. HotpotQA에서 citation F1을 재측정하십시오. HotpotQA는 supporting facts가 데이터셋에
   독립적으로 주어지므로 gold가 순환적이지 않습니다. 6개 시스템 전부, n=100.
3. 두 결과(Full Benchmark vs HotpotQA)를 나란히 놓고, 순위가 유지되는지 보고하십시오.
   유지된다면 헤드라인 주장이 훨씬 강해집니다. 뒤집힌다면 그 사실을 논문에 실어야 합니다.

산출물: docs/paper_revision/B2_citation_metric.md + 재측정 결과

────────────────────────────────────────────────────────
[단계 7] B4 — FlatRAG baseline의 위치 정리 (코드 조사 + 서술 제안)
────────────────────────────────────────────────────────
FlatRAG는 "a hybrid of BM25 60%, semantic 25%, structural 15%"로만 설명되고 인용이 없는
저자 자작 baseline입니다. 그런데 Table 8에서 ROUGE-L 1위(0.479)이고, 논문의 중심 주장인
cautionary finding("평면 검색이 lexical 지표에서 앞선다")의 주 증거입니다.

할 일:
1. src/core/flat_rag_baseline.py에서 60/25/15의 정확한 의미와 출처를 확인하십시오.
   선행연구에서 가져온 값인지, 임의 선택인지.
2. 임의 선택이라면 가중치 sweep을 돌려 결과가 얼마나 민감한지 보이십시오
   (최소 3~4개 설정, fair protocol n=100 또는 비용상 n=50).
3. 논문 서술 제안: cautionary finding을 FlatRAG가 아니라 BM25(표준 baseline, Table 8에서
   ROUGE-L 0.473으로 사실상 동일) 기준으로 다시 쓰는 초안을 작성하십시오. 그러면 자작
   baseline에 핵심 주장이 얹히는 구조가 사라집니다.

산출물: docs/paper_revision/B4_flatrag.md

────────────────────────────────────────────────────────
[마무리] 논문 반영 초안
────────────────────────────────────────────────────────
위 결과를 바탕으로 docs/paper_revision/PAPER_EDITS.md를 작성하십시오. 형식:

  ## 수정 위치: §5.1 Baselines
  ### 현재 문장
  (원문 그대로)
  ### 교체 문장
  (새 문장 — 실제 측정값 포함)
  ### 근거
  (어느 스크립트/결과 파일에서 나왔는지)

docx는 직접 수정하지 마십시오. 사람이 검토한 뒤 반영합니다.

마지막으로, 측정하지 못한 항목이 있으면 "미측정 목록"으로 따로 모아 왜 못 했는지
적어 주십시오. 이것이 논문 Limitations에 들어갑니다.
```

---

## 참고: A그룹 반영 완료 (2026-07-27)

`TreeRAG_TIST_ACM.docx`에 아래 7건이 반영되었습니다. 백업: `TreeRAG_TIST_ACM_backup_pre_roundA_*.docx`

| # | 내용 |
|---|---|
| A1 | Figure 8 캡션이 "FlatRAG uses no retrieval context **by construction**"이라 하고 Table 8 각주는 "true context is **non-zero**"라 하던 모순 해소. FlatRAG는 Pareto frontier 점으로 읽으면 안 된다고 명시 |
| A2 | FlatRAG의 0 토큰에 대한 Table 4·7·8의 서로 다른 3가지 각주를 **하나의 문구로 통일** ("계측 실패에 따른 결측치이며 실제 컨텍스트는 0이 아님") |
| A3 | §4.3의 1.8–3.2 s와 Table 7의 ~1–3 s가 Table 8의 14–119 s와 자릿수로 어긋나던 문제 → 서로 다른 serving 환경의 수치이며 직접 비교 불가임을 양쪽에 명시 |
| A4 | §3.3.4의 "DFS wins on factual and medical"이 논문 스스로 무효라 한 offline proxy에만 근거하던 문제 → fair protocol에서는 Beam이 우세함을 함께 적고, 해당 지침을 "설계 근거이지 검증된 선택 규칙이 아님"으로 격하 |
| A5 | 근거 없는 운영 주장 정리: "cache hit rate exceeding 90% in our production trials" → 계측하지 않았음을 명시 / "production-ready", "production-hardened" → "deployable", "implemented" |
| A6 | Algorithm 1의 top-K 자식 선택과 Equation 1의 임계값 게이트가 **서로 다른 두 메커니즘**임을 명시하고, top-K가 자식마다 LLM 호출을 유발해 DFS 비용의 주범임을 밝힘 (→ B1의 근거가 됨) |
| A7 | §2.3 제목 "Multi-Hop and **Cross-Document** Reasoning" → "Multi-Hop Reasoning" (cross-document 실험 없음). Table 1의 PageTree-RAG **Multi-doc ✓ → ~**, 캡션에 근거 추가 |
