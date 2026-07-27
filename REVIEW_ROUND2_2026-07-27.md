# PageTree-RAG — 2차 검토: 추가 수정점 24건

1차 수정(2026-07-27) 반영본 `TreeRAG_TIST_ACM.docx` 기준. 1차 리뷰(`REVIEW_TIST_2026-07-27.md`)에서 다루지 않은 **새로운** 항목만 정리했습니다.

우선순위: **A** = 지금 바로 고칠 수 있는 정합성 오류 · **B** = 심사에서 반드시 문제될 방법론 공백 · **C** = 누락된 실험/자료 · **D** = 구성·분량

---

## A. 새로 발견한 정합성 오류 (재실험 불필요)

### A1. Figure 8 캡션 ↔ Table 8 각주 정면 모순

- Figure 8 캡션: "FlatRAG uses no retrieval context **by construction**."
- Table 8 각주 ‡: "Context instrumentation could not capture FlatRAG's passage context; **its true context is non-zero**."

같은 데이터를 두고 "설계상 0"과 "측정 실패라 실제로는 0이 아님"이 공존합니다. 둘 중 하나가 사실이어야 하며, 후자가 맞다면 **Figure 8의 Pareto frontier 주장 자체가 성립하지 않습니다**(FlatRAG의 x좌표를 모르므로).

### A2. FlatRAG의 "0 토큰"에 대해 표마다 다른 설명 3가지

| 위치 | 설명 |
|---|---|
| Table 4 ‡ | "offline mode uses no retrieved passage context (0 tokens); online mode retrieves hybrid chunks" |
| Table 7 ‡ | "offline mode uses no retrieved passage context" |
| Table 8 ‡ | "instrumentation could not capture...; true context is non-zero" |

하나의 각주 문구로 통일하고, 어느 표가 offline이고 어느 표가 online(fair)인지 명시해야 합니다.

### A3. 지연시간 수치가 자릿수 단위로 어긋남

- §4.3: "cold LLM call **1.8–3.2 s**", 캐시 히트 ~100 ms
- Table 7 각주: "online generation adds **~1–3 s** per query"
- Table 8 실측: baseline **14–21 s**, PageTree-RAG **94–119 s**

§4.3과 Table 7은 호스팅 API 기준, Table 8은 로컬 8B 기준으로 보입니다. 그렇다면 각각 어느 환경의 수치인지 못박아야 합니다. 지금은 같은 시스템의 지연시간이 1.8초라고도 하고 119초라고도 합니다.

### A4. §3.3.4 Algorithm Selection의 근거가 무효화된 프로토콜

> "Section 5 shows that this qualitative guidance is borne out empirically: DFS wins on single-document factual and medical queries, while Beam Search wins on multi-hop synthesis."

- "DFS wins on factual": Table 11에서 factual 최고 ROUGE-L은 **FlatRAG**(0.615). Table 8에서 DFS는 LLM-Judge 최하위(0.785).
- "DFS wins on medical": Table 4는 **offline extractive** 결과이고, Table 13은 medical vs BM25가 **Δ=+0.007, p=0.63으로 null**이라고 보고합니다.

즉 이 문장의 근거는 논문이 §5.7에서 스스로 "체계적으로 편향됐다"고 선언한 offline proxy뿐입니다. **fair protocol 기준으로 다시 쓰거나, "offline 기준"임을 명시**해야 합니다. 그대로 두면 리뷰어가 "자기가 무효라고 한 결과로 설계 주장을 정당화한다"고 지적합니다.

### A5. §4.3의 근거 없는 운영 수치

> "a cache hit rate exceeding 90% in our **production trials**"

production trial이 무엇인지(기간, 질의 수, 사용자 수) 정의가 없고 데이터도 없습니다. 수치를 뒷받침하거나 문장을 삭제하세요. §1·§4의 "production-ready"라는 표현도 배포 근거가 없으면 "deployment-oriented" 정도로 낮추는 편이 안전합니다.

### A6. Algorithm 1의 자식 선택 메커니즘이 본문과 다름

Algorithm 1: `children <- top-K children by LLM-judged relevance   # K = max_branches`
§3.3.1 본문: 자식은 Equation 1의 **임계값 게이트**로 수락/거부.

top-K 랭킹과 임계값 게이트는 다른 메커니즘입니다. 둘 다 쓰인다면 순서(게이트 후 top-K인지, top-K 후 게이트인지)를 명시하고, top-K 랭킹에 드는 **추가 LLM 호출 비용**도 §B1의 회계에 포함해야 합니다.

### A7. 관련연구 소제목이 다루지 않는 내용을 약속

§2.3 제목 "Multi-Hop and **Cross-Document** Reasoning" — 본문에도 실험에도 cross-document 추론은 없습니다. Table 1에서는 Multi-doc을 ✓로 표시했지만 §5 어디에도 다문서 실험이 없습니다. 제목에서 "Cross-Document"를 빼거나, Table 1의 Multi-doc을 ~로 낮추세요.

---

## B. 심사에서 반드시 문제될 방법론 공백

### B1. ⚠️ 토큰 효율 주장이 회계의 절반만 계산 — 가장 큰 취약점

논문은 **generator에 들어간 컨텍스트 토큰만** 셉니다. 그런데 PageTree-RAG는 질의마다 노드 관련성 채점을 위해 **다수의 추가 LLM 호출**을 합니다(§6.5도 이를 지연시간의 원인으로 인정). 그 호출들이 소비한 토큰은 어디에도 집계되지 않습니다.

- 질의당 LLM 호출 수가 논문 전체에 **한 번도 나오지 않습니다.**
- DFS는 노드당 1회, Beam은 10노드 배치당 1회 호출이므로, b=3·d=5면 수십~수백 회 규모입니다.
- 각 호출은 노드 title+summary+query를 프롬프트로 넣으므로, **파이프라인 전체 토큰 소비는 BM25(1회 호출)보다 압도적으로 클 가능성이 높습니다.**

제목과 초록에 "Token-Efficient"를 넣은 이상, 리뷰어는 반드시 총 비용을 묻습니다. **질의당 LLM 호출 수 / 입력·출력 토큰 / 추정 비용을 담은 표를 추가**하고, 현재의 16토큰 주장을 "generator context"로 한정해 다시 서술하세요. 정직하게 "검색 단계 비용을 생성 단계 컨텍스트와 교환한 것"이라고 프레이밍하면 오히려 방어됩니다.

### B2. ⚠️ Citation F1의 gold 정의가 순환적일 수 있음

Full Benchmark는 "LLM이 생성하고, answer hint가 **source node**에 verbatim으로 나타나는지 검증"해 만들었습니다(§5.1). 그렇다면 gold supporting section = **질문을 생성한 그 노드**입니다. 즉 **정답 단위가 PageTree-RAG의 검색 단위와 동일**합니다.

청크 기반 시스템은 애초에 노드 경계와 일치하지 않는 단위를 반환하므로 구조적으로 불리합니다. 헤드라인 결과(citation F1 = 0.757)가 이 설계 때문일 수 있다는 반박을 막으려면:

1. gold section이 데이터셋에서 **독립적으로** 주어지는 HotpotQA(supporting facts)에서 citation F1을 재측정하거나,
2. 최소한 이 순환성을 Threats to Validity에 명시하고 매칭 규칙(노드↔청크 대응을 어떻게 판정했는지)을 밝히세요.

현재 §6.4의 "Citation metric" 항목은 "provenance fidelity이지 answer correctness가 아니다"까지만 말하고 이 문제는 언급하지 않습니다.

### B3. baseline 하이퍼파라미터가 전무

§5.1 Baselines 문단은 4개 baseline을 각각 한 구절로만 설명하고 **재현에 필요한 값이 하나도 없습니다**:

- chunk size / overlap (BM25, Dense, FlatRAG, RAPTOR 공통)
- top-k
- BM25의 k1, b
- Dense의 Sentence-BERT 체크포인트명
- RAPTOR의 클러스터 수, 요약 모델, 트리 높이

특히 **chunk size는 "10× 적은 토큰" 주장의 분모**입니다. 청크를 크게 잡으면 배수가 커지고 작게 잡으면 사라집니다. 이 값이 없으면 헤드라인 수치를 검증할 수 없습니다.

### B4. FlatRAG가 자작 baseline인데 중심 주장의 주 증거

FlatRAG = "a hybrid of BM25 60%, semantic 25%, structural 15%" — 이 가중치의 출처·근거가 없고 인용도 없습니다. 그런데 이 시스템이 Table 8 ROUGE-L 1위(0.479)이자 cautionary finding("평면 검색이 lexical 지표에서 앞선다")의 핵심 증거입니다.

→ FlatRAG를 저자 구성 baseline임을 명시하고, 가중치 선택 근거(또는 sweep)를 밝히세요. 아니면 cautionary finding을 **BM25**(표준 baseline) 기준으로 다시 쓰는 편이 훨씬 견고합니다. BM25도 Table 8에서 ROUGE-L 0.473으로 사실상 같습니다.

### B5. ROUGE-L / BERTScore 변형 미명시

- "ROUGE-L [21] (**longest-common-subsequence recall**)" — ROUGE-L은 통상 F-measure로 보고합니다. recall만 쓰면 길게 답할수록 유리해집니다.
- "BERTScore [40] (**contextual embedding precision**)" — BERTScore도 통상 F1입니다. precision만 쓰면 짧게 답할수록 유리합니다.

**두 지표가 서로 반대 방향으로 답변 길이에 편향됩니다.** 논문 전체의 논지가 "ROUGE-L 순위 역전"에 걸려 있으므로, 어떤 변형인지·어떤 구현체인지·rescaling 여부·stemming 여부를 명시하고, 가능하면 F 변형을 병기하세요. 시스템별 평균 답변 길이도 함께 보고하면 반박을 원천 차단할 수 있습니다.

### B6. 한국어 프롬프트 × 영어 벤치마크라는 교란변수

Appendix A.2: traversal 프롬프트가 **released implementation에서 한국어**로 작성돼 있습니다. 그런데 HotpotQA·GovReport는 영어이고, §6.3은 생성기가 영어 질문에 한국어로 답하는 버그(100건 중 28건)를 보고합니다.

이 언어 불일치는 **PageTree-RAG 경로에만 존재하는 교란변수**입니다(baseline은 LLM 호출을 하지 않으므로). 다음을 명확히 해주세요:

- 영어 벤치마크 실행 시 traversal 프롬프트도 한국어였는지
- 그렇다면 영어 프롬프트로 바꿨을 때 결과가 어떻게 달라지는지(간단한 대조 실험)
- §6.3의 버그가 traversal 프롬프트에서 온 것인지 생성 프롬프트에서 온 것인지

### B7. ⚠️ 대규모 평가가 실제 시스템이 아니라 fallback 경로

§4.5: "the **offline keyword-scoring path** used for large-scale evaluation (Section 5) doubles as a **fallback** when the LLM API is rate-limited or unavailable."

즉 Table 3(n=204), Table 4(medical n=42), Table 5(ablation n=70), Table 7(n=204), 그리고 Table 13의 **General/Medical 행 전체**가 LLM 노드 채점을 쓰지 않은 **축소 버전**의 결과입니다. 13개 표 중 4개와 통계표의 절반이 여기 해당합니다.

논문의 핵심 기여인 "LLM 기반 적응형 traversal"이 이 결과들에는 **작동하지 않았다**는 뜻입니다. 최소한:

- 해당 표 캡션 전부에 "keyword-only scoring path (no LLM node judgment)"를 명시하고,
- medical benchmark(n=42, 소규모)만이라도 fair protocol로 재실행하는 것을 강력히 권합니다. medical entity recall 1.000은 논문에서 가장 인상적인 수치인데 지금은 fallback 경로 결과입니다.

---

## C. 누락된 실험·자료

### C1. 인덱싱 비용 미보고

"one-time cost"라고만 하고 문서당 **소요 시간·토큰 수·API 비용**이 없습니다. 시스템 논문에서는 필수입니다. 7개 벤치마크 문서 + GovReport 40건 + HotpotQA 100건의 총 인덱싱 비용을 보고하세요.

### C2. 트리 통계 미보고

§3.1은 "d_max 4–6", §3.2는 "42쪽 논문 → 깊이 4, 노드 80–120개"라는 **일화 하나뿐**입니다. 사용한 전 코퍼스에 대해 노드 수·깊이·평균 fan-out·페이지당 노드 수의 분포를 표로 제시해야 traversal 복잡도 논의(O(b^d), O(W·d·f))가 근거를 갖습니다.

### C3. 다문서 라우팅 미평가

§4.2 Stage 2 (1)에 "routes the query to one or more indexed documents"가 있고 Table 1도 Multi-doc ✓인데, 실험은 전부 단일 문서입니다. 라우팅 정확도를 측정하거나 기여 목록에서 빼세요.

### C4. 확장성 실험 없음

문서 수 증가(10 → 100 → 1000)나 문서 길이 증가에 따른 지연시간·품질 거동이 없습니다. §3.3.3의 "메모리가 코퍼스 크기 N과 무관"이라는 주장도 실측이 없습니다.

### C5. 결과 표에 분산·신뢰구간 없음

Table 3·4·5·7·8·10·11·12 전부 점추정치만 있습니다. 최소한 Table 8·10·12에는 표준오차나 부트스트랩 CI를 병기하세요 — 특히 "0.822 vs 0.826 tie"를 주장하려면 필수입니다.

### C6. Medical Benchmark 공개 가능성 확인 필요

"Japanese-language biomedical engineering **lecture materials**"로 만들었는데 §6.7은 "all four evaluation sets"를 공개한다고 합니다. 저작권 문제가 없는지 확인하고, 공개 불가라면 그 사실을 명시하세요.

### C7. 벤치마크 문서 목록 없음

"seven documents (academic papers and biomedical reports)" — 어떤 문서인지, 몇 쪽인지, 어디서 구할 수 있는지 없습니다. 부록에 목록을 넣으세요.

---

## D. 구성·분량 (TIST 25쪽 제한 대응)

### D1. 기여 목록이 §1과 §7에 두 번 (약 1쪽 중복)

§1의 (1)(2)(3)과 §7의 (1)(2)(3)이 거의 같은 내용입니다. **삭제 1순위**입니다. §7에서는 목록을 없애고 한 문단으로 압축하세요.

### D2. 핵심 메시지 5회 반복

"structure buys verifiability and efficiency, not surface-string similarity"가 Abstract, §1 말미, §6.1 첫 문단, §6.1 마지막, §7 마지막 문단에 반복됩니다. 2회로 줄이세요.

### D3. Figure 3·4·7·8이 같은 비교의 변형

넷 다 fair protocol n=100의 시스템 간 비교입니다. Figure 3+4를 하나로, Figure 7+8을 하나(2-panel)로 합치면 최소 1쪽이 절약되고 가독성도 올라갑니다.

### D4. §5.4에 프로토콜이 다른 두 실험이 섞임

Table 5(ablation, **n=70, offline extractive**)와 Table 6(sensitivity, **n=50, fair generative**)이 같은 절에 있습니다. 프로토콜이 다르므로 절을 나누거나, 최소한 각 표를 소개하는 문장에서 프로토콜을 먼저 밝히세요.

### D5. ACM 관례상의 섹션 순서

현재: Conclusion → GenAI Usage Disclosure → APPENDIX → REFERENCES
ACM 관례: Conclusion → **Acknowledgments** → GenAI Disclosure → References → Appendices

**ACKNOWLEDGMENTS 절이 없습니다.** 지도교수·연구비 지원이 있다면 반드시 넣어야 합니다(연구비 명시는 많은 기관에서 의무입니다).

### D6. Equation 3이 언급 순서보다 늦게 등장

§3.3.1이 "computed in closed form (Equation 3, below)"라고 앞서 참조한 뒤 Equation 2가 나오고 그다음 Equation 3이 나옵니다. Equation 3을 τ 설명 직후로 옮기면 읽기 흐름이 자연스러워집니다.

---

## 권장 처리 순서

1. **A1–A7** (반나절) — 문장 수정만으로 끝납니다. 지금 바로 하세요.
2. **B3, B5** (반나절) — 코드에서 값을 꺼내 §5.1에 적으면 됩니다. **투고 전 필수**입니다.
3. **B1** (1–2일) — 로그에서 질의당 LLM 호출 수와 토큰을 집계해 표 하나 추가. 이것 하나로 가장 큰 반박이 막힙니다.
4. **B7 부분 대응** (2–3일) — medical n=42만 fair protocol 재실행.
5. **B2** (2–3일) — HotpotQA에서 citation F1 재측정.
6. **D1–D4** (반나절) — 분량 확보.
7. C 항목은 여유가 되는 대로. C1·C2·C7은 자료 정리만으로 가능합니다.
