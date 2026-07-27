# PageTree-RAG — ACM TIST 투고 대비 종합 검토

> **[2026-07-27 업데이트]** 아래 §5의 "즉시" 항목 9건과 서술 톤 조정이 `TreeRAG_TIST_ACM.docx`에 **모두 반영되었습니다**. 원본은 `TreeRAG_TIST_ACM_backup_pre_review_20260727_140850.docx`(및 동명 .pdf)로 백업했습니다. 반영 내역과 남은 과제는 문서 맨 끝 **§7**을 보세요.

검토 대상: `TreeRAG_TIST_ACM.docx` (본문 316 단락 / 표 13개 / PDF 35쪽)
검토일: 2026-07-27

---

## 0. 총평 (한 문단)

시스템 엔지니어링과 **정직한 서술**은 이 논문의 가장 큰 강점입니다. 특히 (a) generator-controlled 평가 프로토콜, (b) extractive 평가가 순위를 뒤집는다는 cautionary finding, (c) 두 건의 버그(캐시 키 충돌, HotpotQA 빈 컨텍스트)를 스스로 공개하고 재실행한 점은 리뷰어가 높이 평가할 부분입니다. 그러나 현재 상태로 TIST에 내면 **Major Revision 또는 Reject**가 유력합니다. 이유는 세 가지입니다.

1. **헤드라인 주장이 표와 어긋납니다.** Figure 3·7·8 캡션과 Section 5.3 본문이 Table 8·9와 정면으로 모순됩니다 (§2 참조).
2. **"10× 적은 토큰"이라는 핵심 기여의 측정 타당성이 의심됩니다.** 16토큰으로 답을 생성했다는 것은 baseline의 111~222토큰과 같은 단위로 셌다고 보기 어렵습니다 (§3.1).
3. **실증 근거가 저널 분량 대비 얇습니다.** 외부 벤치마크는 HotpotQA(트리가 평면이라 계층성 검증 불가)와 GovReport(n=40)뿐이고, 가장 가까운 경쟁자인 PageIndex는 Table 1에만 있고 실험에는 없습니다 (§3.3).

참고로 TIST는 **참고문헌 포함 25쪽 제한**입니다. 현재 PDF 35쪽 → 조판 형식 확인 필요.

---

## 1. Reference 전수 확인 결과

인용 번호 [1]–[42]는 **모두 본문에서 최소 1회 인용**되어 있고, **모든 문헌이 실재**합니다. 유령 인용(hallucinated reference)은 없습니다. 다만 서지 정보 오류가 6건 있습니다.

### 반드시 고쳐야 할 오류

| # | 문제 | 수정 |
|---|---|---|
| **[23]** | 제목·저자 모두 오류. "A Replication Study of Dense Passage **Retrieval**", 저자 "Ruqing Sun" | → "A Replication Study of Dense Passage **Retriever**", 저자는 Xueguang Ma, **Kai Sun**, Ronak Pradeep, Jimmy Lin |
| **[33]** | PageIndex 연도 **2023** | → **2025** (VectifyAI가 2025년 9월 공개). 2023은 사실과 다름 |
| **[41]** | MemoRAG 저자 순서·누락. "Qian, Peitian Zhang, Zheng Liu, Mao, Dou" | → Hongjin Qian, **Zheng Liu, Peitian Zhang**, Kelong Mao, Defu Lian, Zhicheng Dou, Tiejun Huang (전체 나열) |
| **[10]** | arXiv만 인용 | HiRAG는 **EMNLP 2025 Findings** 게재 확정. 출판본으로 교체 |
| **[3]** | GraphRAG 연도 2025 | arXiv:2404.16130 원본은 **2024**. 연도 통일 필요 |
| **[42]** | 저자 5인 (Vasilakos 포함) | arXiv v1 기준 4인(Singh, Ehtesham, Kumar, Khoei). 인용 버전 명시 필요 |

### ACM Reference Format 관련 (에디터 데스크 리젝 사유가 됨)

- **정렬 붕괴**: [1]–[38]은 제1저자 성 기준 알파벳순인데 **[39]–[42]가 뒤에 덧붙어 순서가 깨짐**. Tao/Huang/Qian/Singh를 알파벳 위치로 재삽입하고 전체 번호를 다시 매겨야 합니다 (본문 인용 번호도 함께 갱신).
- **`et al.` 사용**: [2], [9], [19], [38]. ACM Reference Format은 **전체 저자 나열**을 요구합니다.
- **발음 부호 누락**: Stéphane Clinchant, Hervé Déjean, Hervé Jégou, Wojciech Kryściński, Barlas Oğuz, Cícero Nogueira dos Santos — 모두 accent 없이 표기됨.
- **서식 불일치**: [39], [40]만 `pp.` 사용. DOI가 있는 항목과 없는 항목이 섞임([13], [23], [28] 등 DOI 누락).
- **[12]** FAISS: 연도 2019인데 권/호(7권 3호)는 2021년치. 하나로 통일.

### 인용 내용의 정확성 (내용 검증)

본문 서술과 원논문 내용은 대체로 정확합니다. 다만:

- **[10] HiRAG와 [37]의 이름 충돌**: [37] (Zhang et al., arXiv:2408.11875)의 시스템 이름도 **HiRAG**입니다. 본문에서 [10]만 "HiRAG"라 부르면 혼동됩니다. [37]을 "HiRAG-Rethink" 등으로 구분하거나 이름을 쓰지 마세요.
- **[30] RAPTOR 서술**: "does offer two retrieval strategies—tree traversal and collapsed tree—but these are fixed a priori"는 정확합니다. 다만 Table 1에서 RAPTOR의 Trav.를 ✗로 준 근거를 캡션에 명시해야 반박을 막을 수 있습니다.

---

## 2. 내부 정합성 오류 (가장 시급 — 리뷰어가 즉시 잡습니다)

### 2.1 표와 정면으로 모순되는 서술

| 위치 | 서술 | 실제 표 |
|---|---|---|
| **Figure 3 캡션** | "PageTree-RAG leads on LLM-Judge (Table 8)" | Table 8: **BM25 0.826 > Beam 0.822**. 본문 스스로 "tie"라 함 |
| **Figure 7 캡션** | "PageTree-RAG reaches the highest judged quality" | 동일 모순 |
| **Figure 8 캡션** | "highest judged quality at 10× fewer context tokens" | 동일 모순 |
| **Section 5.3 (Medical)** | "RAPTOR... citation availability = 0.000" | **Table 9: RAPTOR availability = 1.000** |
| **Section 5.7** | "FlatRAG leads ROUGE-L (**0.405**), DFS falls to **0.308**" | **Table 8: 0.479 / 0.340** — 옛 40문항 실행 결과가 남은 것으로 보임 |
| **Section 5.1** | "fair-protocol experiments are run on a fixed **40-question** sample" | Table 8·11: **n = 100** |
| **Section 6.4 (Threats)** | "per-type cells (**n = 8–21**)" | Table 11: **49 / 31 / 20** |
| **Section 6.4** | "citation and per-type analyses of **Tables 7 and 9**" | Table 7은 Efficiency(n=204). → **Tables 9 and 11** |
| **Section 3.5** | 압축이 "22–37% 감소(**Table 4**)" | Table 4에 압축 비교 없음 → **Table 5** |
| **Section 5.10** | "약 20 토큰(DFS 5) vs 약 1,950" | **Table 12에 컨텍스트 열이 아예 없음** |

### 2.2 Table 9의 논리적 모순 (치명적)

FlatRAG: **Citation Availability = 0.000, 그런데 Citation F1 = 0.570.**
페이지 참조를 단 하나도 내놓지 않는 시스템이 인용 F1 0.570을 받는 것은 정의상 불가능합니다. 두 지표의 산식이 서로 다른 대상을 재고 있다는 뜻이며, 이는 논문의 **가장 강한 주장(citation faithfulness)의 근거를 무너뜨립니다**. 두 지표의 정확한 정의(무엇을 gold로, 무엇을 예측으로 보는지)를 수식으로 명시하고 FlatRAG 행을 재계산해야 합니다.

부수적으로 DFS의 availability가 0.875인 것도 "모든 노드가 page range를 보유하므로 항상 인용 가능"이라는 §3.1 주장과 충돌합니다. 12.5%가 왜 실패했는지 설명이 필요합니다.

### 2.3 수식·알고리즘 오류

- **복잡도 오기**: §3.3.2 "The number of nodes evaluated is **O(b · d)**... b=3, d=5 yield at most **243**". 243 = 3⁵ = **b^d**입니다. `O(b·d)` = 15. 뒤(§3.3.3)에서는 "DFS worst case of b^d"라고 옳게 씁니다. → **O(b^d)로 수정**.
- **Equation 2 ≠ Algorithm 2**: Eq.2는 가중합 `0.6·llm + 0.2·kw + 0.2·structural`인데, Algorithm 2는 `child_score ← score × (…)`로 **경로 누적곱**입니다. 곱셈이 실제 구현이라면 Eq.2를 재귀식으로 다시 쓰거나, 최소한 "부모 점수와의 곱으로 전파된다"를 명시해야 합니다. 현재는 정의가 두 개입니다.
- **Eq.1 표기**: `accept(v) = [ … ] > τ`의 대괄호가 불필요/모호. Iverson 괄호인지 단순 그룹인지 불명확.

### 2.4 산술 오류

- §5.5: "RAPTOR uses the most context (218.7 tokens, **+108.7% vs. DFS**)". 218.7 / 65.5 → **+233.9%**. (Beam 85.9 기준으로도 +154.6%.) 숫자 재계산 필요.
- §5.4: "inflates context by **+22.8%** (107.4 to 170.5 tokens)". 107.4→170.5는 **+58.8%**. +22.8%는 Full(138.9) 대비 값입니다. 기준을 하나로 통일하세요.
- §5.2 "+39.5% ROUGE-L (+0.127)" vs Table 13 "Δ = +0.128" — 반올림 불일치(경미).

### 2.5 절 번호 체계 오류 (전면 점검 필요)

Section 6의 소절은 6.1 Why Structure… / 6.2 DFS vs Beam / 6.3 Error Analysis / 6.4 Threats / 6.5 Limitations / 6.6 Broader Impact / 6.7 Reproducibility 순인데, 본문은 버그 논의를 **"Section 6.2"**로 6회 이상 참조합니다(실제 6.3). Threats도 "Section 6.3"으로 참조(실제 6.4). **Section 6 전체에 off-by-one이 걸려 있습니다.**
또한 Table 11 캡션의 "corrected — **Section 4.3**"은 Caching Strategy를 가리킵니다(→ Error Analysis). §5.6/§6.1의 "multi-hop (Section 5.6)"도 실제로는 5.9입니다. 최종 조판 전 상호참조를 필드로 자동화하거나 전수 확인하세요.

### 2.6 Figure 배치

Figure 3·4는 **Section 5.2(extractive)** 문단에 붙어 있는데 내용은 **fair protocol(n=100)** 결과입니다. Figure 7·8도 offline n=204 문단(§5.5)에서 호출되지만 캡션은 fair n=100입니다. 그림을 해당 결과 절로 옮기세요.

---

## 3. 방법론·실험에 대한 리뷰어 관점 지적

### 3.1 "16 토큰" — 가장 큰 리스크

Table 8에서 PageTree-RAG는 **8~16 토큰**으로 답을 생성했다고 보고합니다. 16토큰은 대략 12단어로, MAX_OUTPUT_TOKENS = 4000이라는 압축기 설정과도 3 자릿수 차이가 납니다. 리뷰어는 즉시 "PageTree-RAG는 **노드 title/summary만** 세고, baseline은 **청크 원문 전체**를 센 것 아닌가"를 물을 것입니다. 만약 그렇다면 "an order of magnitude fewer context tokens"는 **동일 단위 비교가 아니며**, 세 가지 헤드라인 기여 중 하나가 무너집니다.

→ 대응: (i) 실제로 generator prompt에 들어간 문자열을 그대로 tokenizer로 센 수치임을 명시하고, (ii) 대표 질의 1건의 **실제 프롬프트 전문**을 부록에 싣고, (iii) 16토큰 컨텍스트로 LLM-Judge 0.822가 나온 것이 검색 덕분인지 **8B 모델의 파라메트릭 지식 덕분인지**를 구분하는 실험(빈 컨텍스트 대조군)을 추가하세요. 이미 §6.3에서 빈 컨텍스트 버그 때 "생성기의 자체 지식으로 답이 나왔다"고 인정했으므로, 이 대조군은 필수입니다.

### 3.2 llm(v,q)를 누가 계산하는가 — 공정성 논란

Appendix A.2: 두 traversal 모두 **gemini-2.5-flash**(temperature 0, JSON 강제)로 노드 관련성을 채점한다고 명시합니다. 반면 §5.1과 §6.7은 "모든 생성 결과는 로컬 Llama 3.1 8B 하나만 사용"이라고 합니다.

**fair generative protocol에서 PageTree-RAG만 검색 단계에 프런티어 모델(Gemini)을 쓴다면, 이는 generator-controlled가 아니라 retriever에 추가 모델이 들어간 비대칭 비교입니다.** 논문의 방법론적 핵심 주장이 여기서 무너질 수 있습니다. §5.1에 "fair protocol의 노드 채점에 사용한 모델은 X"라고 한 문장으로 못박고, 만약 Gemini를 썼다면 Llama 3.1 8B로 채점한 결과를 함께 보고해야 합니다.

### 3.3 비교 대상의 공백

- **PageIndex [33]을 실험에 넣지 않았습니다.** vectorless + 페이지 참조 트리 + 구조 우선 탐색 — 이 논문과 거의 동일한 설계 공간의 유일한 직접 경쟁자입니다. Table 1에서만 언급하고 실험에서 빠진 것은 리뷰어가 반드시 지적합니다. 공개 구현(MIT)이 있으므로 실행 가능한 요구입니다.
- **[39] TreeRAG (Tao et al., ACL 2025 Findings)** 역시 이름까지 같은 시스템인데 실험 비교가 없습니다. 최소한 "코드 미공개/데이터셋 상이로 비교 불가"의 근거를 밝혀야 합니다.
- **RAPTOR baseline**이 공식 구현인지 재구현인지 불명확합니다. Table 3에서 RAPTOR가 ROUGE-L 0.194, Table 4 medical에서 0.053으로 붕괴하는데, 이 정도 수치는 "baseline 구현이 잘못된 것 아닌가"라는 의심을 삽니다. 구현 출처와 하이퍼파라미터를 명시하세요.

### 3.4 FlatRAG baseline의 신뢰성

Table 7 각주 "FlatRAG in offline mode uses **no retrieved passage context (0 tokens)**", Table 8 각주 "context instrumentation **could not capture** FlatRAG's context". 그런데 이 FlatRAG가 **Table 8에서 ROUGE-L 1위(0.479)**이고, 논문의 중심 주장인 "cautionary finding"(평면 검색이 lexical 지표에서 앞선다)의 **주 증거**입니다. 계측조차 안 되는 baseline 위에 핵심 방법론적 결론이 얹혀 있는 구조는 방어하기 어렵습니다. 계측을 고치거나, cautionary finding을 BM25 기준으로 재서술하세요.

### 3.5 Table 6의 자기모순 (하이퍼파라미터 민감도)

같은 **기본 설정**(W=5, D=5, 0.6/0.2/0.2, τ=0.7)이 네 sweep에서 각각 다르게 나옵니다.

| sweep | ROUGE-L | LLM-Judge | Ctx | Latency |
|---|---|---|---|---|
| Beam width 행 | 0.274 | 0.813 | 14.1 | 87.2 |
| Max depth 행 | 0.299 | 0.787 | 15.4 | 102.2 |
| Weights 행 | 0.267 | 0.805 | 15.4 | 112.4 |
| Compression 행 | 0.242 | 0.807 | 13.8 | 95.6 |

즉 **동일 조건의 run-to-run 변동이 ROUGE-L 0.057, LLM-Judge 0.026**입니다. 이 노이즈가 sweep에서 관찰된 대부분의 차이(예: τ=0.8이 0.7을 0.025 앞섬)보다 **큽니다**. 따라서 "no single hyperparameter is a hidden lever"라는 결론은 현재 데이터로 지지되지 않습니다.

→ 대응: 설정당 3~5 seed 반복 + 표준편차 병기, 또는 이 변동성을 명시적으로 보고하고 "차이가 노이즈 범위 내"라고 서술 변경. 부록 A.1에서 judge temperature를 고정하지 않았다고 인정한 부분과도 직결됩니다 — **judge temperature를 0으로 고정하고 재실행**하는 것이 가장 깔끔합니다.

추가로 Beam width W=1의 latency 56.1s > W=3의 10.7s는 단조성이 깨져 있어 설명이 필요합니다.

### 3.6 Table 1의 과잉 주장

- PageTree-RAG의 **Trav.(query-time choice among traversal strategies) = ✓**인데, 논문은 §6.5에서 자동 선택기가 **작동하지 않아 배포하지 않았다**고 명시합니다. 표가 논문이 철회한 기능을 주장하고 있습니다 → **~ 또는 ✗**로 수정하거나 열 정의를 "두 정책 제공"으로 바꾸세요.
- **Online(증분 인덱싱) = ✓**도 논문 어디에서도 평가되지 않았습니다.
- Halluc./Compr. 열에서 모든 경쟁자가 ✗인 것은 LlamaIndex의 내장 postprocessor·response evaluator를 고려하면 과합니다.

### 3.7 평가되지 않은 기여

**Grounding confidence module**은 contribution #2의 절반인데, 정확도가 전혀 평가되지 않았습니다(경고 임계값 0.6, 70%만 제시). 최소한 소규모 human-labeled hallucination set에서 precision/recall/AUC를 보고해야 "기여"로 인정됩니다.
**Cross-reference resolution**도 "§5.4의 ablation이 precision/coverage trade-off를 정량화한다"고 했지만 Table 5에는 ROUGE-L delta만 있습니다.

### 3.8 벤치마크 구성의 순환성

Full Benchmark(n=204)는 LLM이 생성하고 "answer hint가 source node에 **verbatim**으로 나타나는지"로 검증했습니다. 즉 **참조 답안 자체가 추출적**입니다. 논문은 §6.1에서 바로 이것이 ROUGE 아티팩트의 원인이라고 스스로 설명합니다. 그렇다면 같은 벤치마크의 ROUGE-L 수치를 주요 결과로 제시하는 것은 일관성이 없습니다.
→ Full Benchmark는 citation F1 / context / LLM-Judge 전용으로 쓰고, ROUGE는 외부 벤치마크에만 쓰는 편이 논지가 훨씬 깔끔해집니다.

또한 HotpotQA는 논문 스스로 "flat, single-level tree"라 계층성 검증이 불가하다고 인정했으므로, **계층성을 실제로 검증하는 외부 데이터는 GovReport(n=40) 하나뿐**입니다. 저널 논문으로는 부족합니다. QASPER, ContractNLI, LegalBench, MultiHop-RAG, 또는 실제 규정 문서 세트를 최소 1개 추가 권장.

### 3.9 통계

- Abstract·Conclusion의 "multi-hop and comparative questions (0.837 and 0.830, **statistically tied** with BM25)" — 이 tie 검정 결과가 논문 어디에도 없습니다. Table 11은 Beam의 LLM-Judge만 싣고 BM25 값은 캡션에만 있습니다. **Table 11에 전 시스템 × 전 지표를 싣고 p값을 병기**하세요.
- Table 13에 Full Benchmark **LLM-Judge 행이 없습니다**. 헤드라인 주장(0.822 tie)이 robust statistics 표에서 빠져 있는 것은 의도적 누락으로 읽힐 수 있습니다.
- Table 10에서 PageTree-RAG 두 변형이 **ROUGE-L 6개 시스템 중 최하위**(0.141/0.145)라는 점은 정직하게 보고되었으나, Abstract에서는 언급되지 않습니다. Abstract에 한 절 추가를 권합니다(정직성이 이 논문의 자산이므로 오히려 유리).

---

## 4. 형식·서식

- **분량**: PDF 35쪽. TIST는 **참고문헌 포함 25쪽**. `acmart` 2단 `journal=TIST` 템플릿으로 조판해 실제 쪽수를 확인하세요.
- **Abstract 약 350단어** — 과도하게 깁니다. 200~250단어로 압축 권장(수치 나열을 2개로 줄이면 됩니다).
- **대시 표기 혼용**: "hallucination-generating plausible…"처럼 em dash 자리에 hyphen이 들어간 곳과 "—"가 제대로 들어간 곳이 섞여 있습니다. 전수 치환 필요.
- **단독 저자인데 "the authors"** (GenAI Usage Disclosure) → "the author".
- **누락 요소**: ACM Reference Format 블록, Acknowledgments, 자금 지원 명시, (권장) 코드/데이터 URL. §6.7에서 "release the full pipeline"이라 했으나 **실제 저장소 링크가 논문에 없습니다** — 재현성 주장에 치명적입니다.
- Table 3 캡션 "Best per column in bold" — 추출본에서 굵게 처리가 확인되지 않습니다. 조판 시 확인.
- Table 3의 Dense/DFS HotpotQA 열 "–"(미측정)에 대한 설명이 캡션에 없습니다.

---

## 5. 우선순위 수정 목록

**즉시 (제출 전 필수)**

1. Figure 3·7·8 캡션을 Table 8에 맞춰 수정 ("leads" → "is statistically tied with the best baseline").
2. §5.3의 "RAPTOR citation availability = 0.000" 삭제/수정 (Table 9는 1.000).
3. §5.7의 0.405 / 0.308 → Table 8 값(0.479 / 0.340)으로 교체.
4. §5.1의 "40-question" → 100 (또는 어떤 표가 40이고 어떤 표가 100인지 명확히 분리 서술).
5. Table 9 FlatRAG 행(availability 0.000 / F1 0.570) 재계산 + 두 지표 수식 명시.
6. 복잡도 O(b·d) → O(b^d), §5.5의 +108.7% 재계산, §5.4의 +22.8% 기준 통일.
7. Section 6 상호참조 off-by-one 전수 수정, Table 11 캡션의 "Section 4.3" 수정.
8. Reference [23] 제목·저자, [33] 연도, [41] 저자 수정 / [39]–[42] 알파벳 순 재삽입 / `et al.` 제거 / accent 복원.
9. 코드·데이터 저장소 URL 삽입.

**Major Revision 수준 (심사 통과를 위해)**

10. 컨텍스트 토큰 측정 방법을 명시하고, 실제 프롬프트 예시 + 빈 컨텍스트 대조군 추가 (§3.1).
11. fair protocol에서 llm(v,q)를 채점한 모델을 명시하고, 필요 시 로컬 모델로 재실행 (§3.2).
12. **PageIndex를 baseline에 추가** (§3.3).
13. Table 6를 seed 3~5회 반복 + 표준편차로 재작성, judge temperature 0 고정 (§3.5).
14. Table 1에서 PageTree-RAG의 Trav./Online 표시 하향 조정 (§3.6).
15. Grounding confidence module의 검출 성능 평가 추가 (§3.7).
16. 계층 구조가 실재하는 외부 벤치마크 1개 이상 추가 (§3.8).
17. Table 11을 전 시스템 × 전 지표 + p값으로 확장, Table 13에 Full Benchmark LLM-Judge 행 추가 (§3.9).

---

## 6. 잘하고 있는 점 (유지할 것)

- Generator-controlled 프로토콜의 설계와, 그 결과가 자기 시스템에 불리하게 나왔을 때도 그대로 보고한 태도.
- 두 건의 버그를 원인·증상·수정·재실행까지 서술한 §6.3 — 리뷰어 신뢰를 크게 얻는 부분입니다.
- 자동 traversal selector가 실패했음을 "future work"로 포장하지 않고 명시적으로 철회한 점.
- Table 13의 bootstrap CI + permutation test + power/n₈₀ 구성. 소표본 논문에서 모범적입니다.
- 부록 A.1/A.2의 프롬프트 전문 공개.

이 세 가지(정직한 프로토콜, 버그 공개, 실패한 컴포넌트의 철회)는 그대로 두고, 위의 정합성 오류와 토큰 측정 문제만 정리하면 논문의 설득력이 크게 올라갑니다.

---

## 7. 적용 완료 내역 (2026-07-27)

`TreeRAG_TIST_ACM.docx` 원본을 직접 수정했습니다. 총 55개 지점 변경. 백업: `TreeRAG_TIST_ACM_backup_pre_review_20260727_140850.docx` / `.pdf`

### 7.1 표–본문 모순 (13건)

- Figure 3·7·8 캡션의 "PageTree-RAG leads / highest judged quality" → "statistically tied with the strongest baseline"
- Figure 3 캡션 참조 Section 5.5 → **5.7**, Figure 4 캡션 Section 5.6 → **5.9**, Figure 4의 "just as it does on the Full Benchmark" → Full Benchmark에서는 동률임을 명시
- §5.3 "RAPTOR citation availability = 0.000" → Table 9와 일치하도록 "citation F1 = 0.000"으로 수정, Figure 5 캡션도 동일 처리
- §5.7 stale 수치 **0.405 → 0.479**, **0.308 → 0.340**
- §5.1 "40-question sample" → **100-question** (Table 9만 40문항임을 괄호로 명시)
- §6.4 "Tables 7 and 9" → **Table 9**, "per-type cells (n = 8-21)" → **Table 11 (n = 20-49)**
- §3.5 압축 효과 참조 Table 4 → **Table 5**
- §5.10 GovReport 토큰 수치가 Table 12에 없음을 명시 + "PageTree-RAG 변형 전체가 우세" → **Beam만 우세, DFS는 BM25·FlatRAG에 뒤짐**으로 정정

### 7.2 수식·산술 (4건)

- 복잡도 **O(b · d) → O(b^d) in the worst case**
- §5.5 RAPTOR 컨텍스트 **+108.7% → +233.9%**
- §5.4 **+22.8%**의 기준을 Full system으로 명시 + base DFS 대비 +58.8% 병기
- Equation 2와 Algorithm 2의 불일치 해소 — "Eq.2는 노드 자체 기여이며 구현에서는 경로를 따라 곱셈으로 전파된다" 문장 추가

### 7.3 상호참조 (16건)

- Section 6 전체 off-by-one 정정: 버그 논의 **6.2 → 6.3** (5곳), Threats **6.3 → 6.4**
- Limitations 참조를 모두 **6.5**로 구체화 (6곳), Reproducibility를 **6.7**로 (2곳)
- 멀티홉 참조 **5.6 → 5.9** (2곳), Table 11 캡션 **"Section 4.3" → "Section 6.3"**
- ※ §6.3 본문의 "response cache key (Section 4.3)"는 Caching Strategy를 가리키는 **정상 참조**라 그대로 두었습니다.

### 7.4 참고문헌 (전면 재작성)

- **알파벳순 재정렬 후 [1]–[42] 재번호 부여**, 본문·표의 인용 24개 단락 자동 갱신 (전수 검증: 42개 모두 인용됨, 번호 연속성 OK)
- [23]→**[24]** 제목 `Retriever`, 저자 `Kai Sun` 정정 / [33]→**[37]** PageIndex 연도 **2025** / [41]→**[28]** MemoRAG 저자 순서·전체 나열 / [10]→**[9]** HiRAG **EMNLP 2025 Findings** 반영 / [3] GraphRAG **2024** / [42]→**[34]** Singh 저자 4인
- `et al.` 4건 → 전체 저자 나열 (Brown, Huang, Lewis, Zheng)
- 발음 부호 복원: Stéphane, Hervé Déjean, Hervé Jégou, Kryściński, Oğuz, Cícero, Küttler, Rocktäschel
- DOI 보강([30] BM25), arXiv ID 보강([11] GovReport, [14] DPR), `pp.` 표기 통일

### 7.5 서술 톤 (10건)

- **Abstract 352 → 252 단어**로 압축, citation F1을 RAPTOR(0.000)뿐 아니라 **최강 baseline(0.562)과도 병기**. Conclusion도 동일 처리
- **Context Tokens 측정 방법 명시** — 어떤 문자열을 셌는지, PageTree-RAG와 baseline이 "종류가 다른" 페이로드임을 밝히고 "정보량의 동일 조건 비교가 아니다"라고 선언 (§3.1 지적 대응)
- **Table 9 캡션에 두 인용 지표의 정의 추가** — Availability는 생성된 답변 텍스트, F1은 검색된 근거의 섹션 식별자를 대상으로 함을 밝혀 FlatRAG의 (0.000, 0.570) 조합을 설명. ⚠️ **이 설명이 실제 구현과 맞는지 코드로 반드시 확인하세요.** 다르면 캡션을 실제 정의로 교체해야 합니다.
- **Table 6 재현 변동성 공개** — 동일 기본 설정이 네 sweep에서 LLM-Judge 0.787–0.813 / ROUGE-L 0.242–0.299로 갈리므로 "0.03 미만 차이는 실재로 읽지 말 것", "beam width만 이 대역을 넘는다"고 명시
- **Table 1** PageTree-RAG의 Trav.·Online을 **✓ → ~**로 하향, 캡션에 근거 명시
- 교차참조 해소 모듈의 "precision/coverage trade-off를 정량화" 주장 철회
- Limitations에 **grounding confidence module의 검출 정확도가 미평가**임을 명시
- Reproducibility에 `<<REPOSITORY URL - TO BE INSERTED>>` 자리표시자 삽입 — **제출 전 반드시 채우세요**
- 단독 저자에 맞춰 "the authors" → "the author"
- Table 3 캡션에 "–" 셀의 의미 설명 추가

### 7.6 남은 과제 (재실험 필요 — 미반영)

| 항목 | 필요 작업 |
|---|---|
| Table 9 FlatRAG 행 | 두 지표 산식을 코드에서 확인하고, 필요 시 재계산 |
| 컨텍스트 토큰 | 실제 프롬프트 전문 부록 수록 + **빈 컨텍스트 대조군** 실험 |
| llm(v,q) 채점 모델 | fair protocol에서 Gemini를 썼는지 명시, 썼다면 Llama 8B로 재실행 |
| PageIndex baseline | 실험에 추가 (공개 구현 있음) |
| Table 6 | seed 3–5회 반복 + 표준편차, judge temperature 0 고정 |
| Table 11 | 전 시스템 × 전 지표 + p값으로 확장 |
| Table 13 | Full Benchmark LLM-Judge 행 추가 |
| Grounding module | labeled hallucination set에서 precision/recall 측정 |
| 외부 벤치마크 | 계층 구조가 실재하는 데이터셋 1개 이상 추가 |
| 분량 | acmart 2단 TIST 템플릿으로 조판해 25쪽 이내 확인 |
| 대시 표기 | 본문 전반의 hyphen/em-dash 혼용 일괄 정리 (Abstract만 정리 완료) |
