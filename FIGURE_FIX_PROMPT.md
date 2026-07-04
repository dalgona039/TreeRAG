# PageTree-RAG — 그림 재생성/수정 실행 프롬프트 (Claude Code)

> 목적: 논문에 넣을 그림 4종의 세 가지 결함을 고쳐 **재생성**한다.
> (1) 모든 그림이 아직 옛 이름 **"TreeRAG"** → **"PageTree-RAG"** 로 개명 반영,
> (2) **efficiency 그림이 본문 주장과 모순**(Latency vs ROUGE-L → TreeRAG가 Pareto에 지배당함) →
>     축을 **context tokens vs LLM-Judge** 로 교체,
> (3) **아키텍처 그림에 Compression·Verification 단계 누락** → 7단계로 보강.
> 대상 파일: `scripts/plot_results.py` (주), `scripts/generate_acm_outputs.py` (텍스트 라벨).
> 출력: `data/benchmark_reports/figures/`.

## 절대 규칙
1. 수치는 산출 JSON(`online_local_llama_general_v3_n100.json`, `..._govreport.json`)에서만 인용. 날조 금지.
2. 편집 후 **반드시 재생성하고 PNG를 눈으로 확인**한다. 어떤 그림에도 "TreeRAG" 문자열이
   남으면 안 된다(제목·범례·축·주석 포함). 확인 방법은 §4.
3. 시스템 표기는 논문과 일치: 시스템명 **PageTree-RAG**, 변형 **PageTree-RAG (DFS)** /
   **PageTree-RAG (Beam)**. (플롯 폭이 좁아 라벨이 겹치면 `PageTree-DFS`/`PageTree-Beam` 축약 허용,
   단 한 그림 안에서는 한 형식으로 통일.)

---

## Fix 1 — 개명 (전 그림 라벨/제목)

`scripts/plot_results.py` 에서 아래를 모두 교체:

- **L452–453, L555** 및 그 밖의 라벨 dict(`_ACM_LABELS`, `LABELS`, `SYSTEM_LABELS` 등):
  `"treerag_dfs": "TreeRAG-DFS"` → `"treerag_dfs": "PageTree-RAG (DFS)"`,
  `"treerag_beam": "TreeRAG-Beam"` → `"treerag_beam": "PageTree-RAG (Beam)"`.
- **L342, L347** 히스토그램 라벨 `'TreeRAG'` / `f'TreeRAG Mean: ...'` → `'PageTree-RAG'`.
- **L400** 제목 `"TreeRAG vs Baselines"` → `"PageTree-RAG vs Baselines"`.
- **L480** `"System comparison: ROUGE-L vs BERTScore"` (해당 없으면 무시).
- **L607** 아키텍처 제목(아래 Fix 3에서 함께 교체).

`scripts/generate_acm_outputs.py` 의 텍스트 스니펫(L56–89 등)의 `TreeRAG`/`TreeRAG-Beam` →
`PageTree-RAG`/`PageTree-RAG (Beam)` (그림은 아니지만 산출 일관성).

> 팁: `grep -rn "TreeRAG" scripts/plot_results.py scripts/generate_acm_outputs.py` 로 잔여 확인.
> 단, RAPTOR 참조나 옛 파일명 문자열은 건드리지 말 것.

---

## Fix 2 — efficiency 그림 축 교체 (가장 중요)

**문제:** 현재 `figure_3_efficiency` 는 `x=a["latency"]`, `y=a["rouge_l"]`(L511–522)라
n=100의 부풀려진 지연(TreeRAG 145–159 s) 때문에 **PageTree-RAG가 Pareto 프론티어 바깥**으로
나오고, 본문의 "적은 context 토큰으로 높은 judged 품질" 주장과 **모순**된다.

**수정:** 이 그림을 **효율=context tokens, 품질=LLM-Judge** 로 다시 그린다.
- `x = a["context_tokens"]` (낮을수록 좋음), `y = a["llm_judge"]` (높을수록 좋음).
- Pareto 프론티어 = **좌상단 지배**(적은 토큰 + 높은 judge). `_pareto_frontier`의 정렬/부등호를
  이 방향(x 최소·y 최대)으로 맞춘다.
- **FlatRAG 주의:** `context_tokens == 0` 은 "검색 컨텍스트 없음(‡)" 아티팩트다. 0을 실제
  최소값으로 취급하면 프론티어를 왜곡한다 → FlatRAG는 마커만 표시하고 **프론티어 계산에서 제외**
  하거나 주석 "no retrieval context (N/A)"로 처리.
- 결과적으로 **PageTree-RAG (Beam) (≈16 tok, judge 0.823)** 와 **(DFS) (≈8 tok, 0.792)** 가
  좌상단 프론티어에 오도록 한다. 제목: `"Efficiency: context tokens vs LLM-Judge — fewer tokens, higher quality (upper-left dominates)"`.
- 파일명은 그대로 `figure_3_efficiency.{pdf,png}` 유지.

**선택(권장):** 지연은 PageTree-RAG의 *비용* 축이므로, 별도 패널
`figure_3b_latency.{pdf,png}` 로 `x=latency(s)`, `y=llm_judge` 를 **정직하게** 그리고
캡션에 "지연이 트리 순회의 비용임"을 명시. (본문 6.x 한계와 일치)

---

## Fix 3 — 아키텍처 그림 7단계로 보강

`figure_architecture` (약 L585)의 `stages` 리스트를 5→**7 박스**로 확장하고 x 좌표 재배치:

```python
stages = [
    ("PDF\nDocument",              0.04, CB_PALETTE[0]),
    ("Zero-shot LLM\nTree Indexer",0.20, CB_PALETTE[1]),
    ("Page-referenced\nStructure Tree", 0.36, CB_PALETTE[2]),
    ("DFS / Beam\nTraversal",      0.52, CB_PALETTE[3]),
    ("Contextual\nCompression",    0.68, CB_PALETTE[4]),
    ("Generation\n(shared LLM)",   0.83, CB_PALETTE[0]),
    ("Hallucination\nVerification",0.96, CB_PALETTE[1]),
]
```
- 화살표 루프(len-1)는 그대로 두면 7박스에 맞게 자동 연결됨. 박스 폭/글자 크기는 겹치면 축소.
- 마지막 산출은 인용 포함을 강조: Verification 다음 텍스트나 캡션에 "→ Grounded answer with [doc, p.X] citation" 명시(박스 추가 대신 캡션도 가능).
- **제목(L607)** → `"PageTree-RAG: structure-preserving, citation-grounded retrieval pipeline"`.

---

## Fix 4 — 재생성 & 검증

```bash
# 그림 생성 엔트리포인트 확인 후 실행 (v3 n=100 fair 리포트 사용)
grep -n "__main__\|def main\|figure_architecture(\|figure_main_bars(\|plot_efficiency" scripts/plot_results.py
python scripts/plot_results.py \
  --report data/benchmark_reports/online_local_llama_general_v3_n100.json
# (엔트리포인트 인자명이 다르면 --help 로 확인해 맞춘다)
```
검증:
```bash
# 1) 스크립트에 잔여 옛 이름 없는지
grep -rn "TreeRAG-\|\"TreeRAG\"\|TreeRAG:" scripts/plot_results.py && echo "‼ 잔여 있음" || echo "clean"
# 2) 생성 PNG를 하나씩 열어 육안 확인: 제목/범례/축에 PageTree-RAG로 표기?
#    figure_1_architecture(7단계+Compression/Verification), figure_2_main_results,
#    figure_3_efficiency(축=tokens vs LLM-Judge, PageTree-RAG가 좌상단 프론티어), figure_3_multihop
```
수용 기준: (a) 4개 그림 어디에도 "TreeRAG" 없음, (b) efficiency 그림에서 PageTree-RAG가
프론티어 위, (c) 아키텍처에 Compression·Verification 단계 존재.

---

## 참고 (그림 아님, 별도 처리)
- `data/benchmark_reports/online_local_qwen_judge_n100.json` 의 `llm_judge` 가 전부 **null**
  (judge_model=null). 실제 교차검증 점수는 `p1c_judge_crossval.json`(ρ=0.60, p=0.208)에 있음.
  이 요약본으로 표/그림을 재생성하면 judge 열이 빈다 → 필요 시 judge를 다시 채워 저장할 것.
- P1-C 정확한 서술: **두 judge 모두 RAPTOR 최하위**, PageTree-RAG (Beam)은 **top-2**
  (llama 1위, qwen 2위·FlatRAG가 qwen 1위), **ρ=0.60이나 n=6이라 p=0.208로 유의하지 않음.**
  "두 judge 모두 Beam 1위"는 부정확하니 쓰지 말 것.
