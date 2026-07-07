> **⚠️ 상위 버그 발견으로 대체됨 (2026-07-07):** 아래 Task A/C가 재실행한
> HotpotQA n=100 데이터 자체가 `_normalize_item()` 파싱 버그로 **전부 빈
> 컨텍스트(children=0)**였음이 추가로 밝혀졌다. 캐시 키 수정(이 프롬프트)은
> 여전히 유효하고 필요하지만, Task A/C의 실행 결과와 Task F가 요구하는 재실행은
> `HOTPOTQA_EMPTY_CONTEXT_FIX_PROMPT.md`로 대체한다 — 그쪽을 먼저 실행할 것.
> Task B(category breakdown, general 벤치마크 기반)는 이 버그와 무관하므로 그대로
> 유효.

# PageTree-RAG — 멀티홉 벤치마크 캐시-오염 재실행 프롬프트 (Claude Code)

> 배경: `scripts/check_cache_contamination.py` 진단 결과, 멀티홉(HotpotQA) 벤치마크
> 리포트 4개(06-28, 06-29, 06-30, 07-06 실행분)가 **전부 100% 오염**되어 있었다.
> `treerag_dfs`와 `treerag_beam`의 `context_tokens`, `retrieved_count`,
> `rouge_l`(소수점 16자리까지)이 완전히 동일 — 서로 다른 두 알고리즘이 같은 캐시
> 객체를 돌려받았다는 뜻이다. 논문의 **Table 10**(문서 내 실제 표, "HotpotQA
> multi-hop (n = 100)")이 이 중 가장 최근 파일
> (`exp2_multihop_hotpotqa_20260706_070918.json`)과 소수점까지 정확히 일치함을
> 확인했다 — 즉 Table 10은 **DFS와 Beam을 비교한 적이 없다.** 이 프롬프트는 이미
> 고친 캐시 버그(`traversal_algorithm`을 cache key에 포함)로 멀티홉 벤치마크를
> 깨끗하게 재실행하고, 관련 표/서술을 실측치로 교체하는 것이 목표다.

## 절대 규칙

1. 수치 날조 금지. 재실행 결과 JSON에서만 논문 수치를 인용한다.
2. 재실행 전 반드시 `python scripts/check_cache_contamination.py`로 **현재 코드가
   실제로 고쳐졌는지** 먼저 확인한다 (이전 auto-policy 재실행에서 42/50→2/50로
   확인된 것과 같은 방식). 새로 만드는 파일도 재실행 직후 같은 스크립트로 다시
   검증해서 오염이 없는지 (exact-match가 우연 수준, 대략 <10%인지) 확인한다.
3. docx 편집 전 백업, Word는 닫고 진행.
4. seed=42 고정.
5. 기존 오염된 파일들(`exp2_multihop_hotpotqa_20260628_204500.json`,
   `..._20260629_133031.json`, `..._20260630_021448.json`,
   `..._20260706_070918.json`, `online_local_llama_general.json`,
   `online_local_llama_general_v2.json`)은 **삭제하지 말고** 파일명에 `_CONTAMINATED`
   를 붙여 보관한다 (근거 자료로 남겨둠). 새 결과는 새 타임스탬프로 저장.

---

## Task A — 멀티홉(HotpotQA) 벤치마크 클린 재실행 → Table 10 교체

1. 현재 코드에 캐시 키 수정이 실제로 반영돼 있는지 확인:
   ```bash
   grep -n "traversal_algorithm" src/core/reasoner.py | grep -i cache
   ```
2. 재실행 (기존 논문이 인용한 것과 동일 설정: n=100, seed=42, local llama3.1:8b +
   local judge):
   ```bash
   python benchmarks/run_exp2_multihop.py \
     --n 100 --seed 42 --skip-partB
   ```
   (Part A만; Part B는 Task B에서 별도로 정리한다. 실제 스크립트가 online
   generation에 필요한 `--gen-backend`/`--gen-model` 류 플래그를 받는지
   `python benchmarks/run_exp2_multihop.py --help`로 먼저 확인하고, 논문이
   써온 것과 동일한 local llama3.1:8b + local judge 설정을 명시적으로 지정할 것.)
3. 결과 파일명을 확인하고 (`exp2_multihop_hotpotqa_<timestamp>.json` 패턴일 것),
   즉시 오염 여부 재검증:
   ```bash
   python scripts/check_cache_contamination.py data/benchmark_reports/exp2_multihop_hotpotqa_<new_timestamp>.json
   ```
   exact-match %가 낮아야(대략 <10%) 정상. 여전히 높으면 캐시 키 수정이 이
   실행 경로에 실제로 타는지 다시 확인 (예: 이 스크립트가 `TreeRAGReasoner`를
   직접 쓰는지, 아니면 별도 캐시를 우회하는 다른 경로를 쓰는지).
4. **보고:** 새 파일의 `summary`에서 시스템별 ROUGE-L / LLM-Judge (BM25, Dense,
   FlatRAG, RAPTOR, PageTree-RAG DFS, PageTree-RAG Beam) — 이게 새 Table 10.
   기존 Table 10과 나란히 비교해서 얼마나 달라졌는지도 알려줄 것 (예: 기존엔
   DFS==Beam=0.077이 "tie"였는데 실제로는 어느 쪽이 더 높은지).

## Task B — Per-type breakdown (Table 11) 재검토

`run_category_breakdown()`(같은 스크립트의 Part B)의 기본 `--existing-report`가
**`online_local_llama_general_v2.json`을 가리키는데, 이 파일도 85% 오염으로
확인됨.** 반면 `online_local_llama_general_v4_n100.json`은 6-7%로 깨끗하고
Table 8/9가 이미 이 파일을 인용하고 있다 (n=100, v2보다 표본도 큼).

1. 클린 파일로 다시 실행:
   ```bash
   python benchmarks/run_exp2_multihop.py --skip-partA \
     --existing-report data/benchmark_reports/online_local_llama_general_v4_n100.json
   ```
2. 산출되는 category-breakdown 결과가 기존 Table 11(질문 유형별 ROUGE-L/Judge)과
   수치가 다른지 비교한다. 다르면 오염이 실제로 Table 11에도 영향을 줬다는 뜻이니
   새 수치로 교체 대상; 우연히 거의 같으면 그대로 둬도 되지만 어느 쪽이든 **어떤
   파일에서 나온 수치인지 근거를 남길 것.**
3. **보고:** 새 category-breakdown 표 전체 (질문 유형, n, 시스템별 ROUGE-L/Judge).

## Task C — Discussion §6.2 Error Analysis 사례 재검증

현재 논문 문장 두 개가 오염된 `exp2_multihop_hotpotqa_20260706_070918.json`의
`per_question` 데이터에서 뽑은 구체적 사례를 인용한다:

- "Are John O'Hara and Rabindranath Tagore the same nationality?" — PageTree-RAG
  (Beam) 0.93 vs BM25 0.20이라는 서술.
- "One Night in Bangkok" 브리지 질문 — PageTree-RAG (Beam)이 틀리고 Dense
  Retrieval이 맞다는 서술.

Task A의 새 클린 데이터에서 같은 `question_id`(있다면) 또는 같은 질문 텍스트를
찾아 값이 그대로인지 확인한다:
```bash
python - <<'PY'
import json
old = json.load(open("data/benchmark_reports/exp2_multihop_hotpotqa_20260706_070918.json"))
new = json.load(open("data/benchmark_reports/exp2_multihop_hotpotqa_<new_timestamp>.json"))
for keyword in ["O'Hara", "Bangkok"]:
    for sysname, records in old["per_question"].items():
        for r in records:
            if keyword.lower() in (r.get("question") or "").lower():
                print(sysname, r.get("question_id"), r.get("llm_judge"), (r.get("answer") or "")[:80])
PY
```
같은 `question_id`를 새 파일에서 찾아 judge 점수/답변이 바뀌었는지 확인.
**바뀌었으면 그 사례는 논문에서 쓸 수 없다** — 새 클린 데이터에서 비슷한 취지의
(맞는 comparison-question 사례, 틀리는 bridge-question 사례) 다른 예시를 찾아
대체하거나, 일반화된 통계적 서술로 대체한다.

## Task D — 캐시 버그 자체를 논문에 투명하게 기록

논문은 이미 §6.2에서 언어-드리프트 버그를 발견/수정/재실행한 과정을 그대로
공개한 전례가 있다 ("Our first pass surfaced a generator bug... We fixed the
bug... and the corrected results... are what we report throughout."). 같은
방식으로 이번 캐시 키 버그도 §4.3(Caching Strategy) 또는 §6.2/Reproducibility에
1-2문장으로 추가할 것을 제안한다:

> "During additional experimentation we discovered that the response cache key
> did not include the traversal algorithm, causing DFS and Beam Search queries
> for the same question to occasionally return each other's cached answer
> within the cache TTL. We fixed the cache key (Section 4.3) and re-ran all
> affected benchmarks; Table 10 and the case studies of Section 6.2 reflect
> the corrected runs."

정확한 문구는 Task A-C 결과가 나온 뒤, 실제로 몇 개 표가 영향을 받았는지에 맞춰
다시 쓴다.

---

## Task E — 논문(docx) 반영

Task A-D 결과가 모두 나온 뒤 진행. Word 닫고 백업 먼저.

1. Table 10 수치를 새 클린 결과로 교체. 캡션의 "DFS and Beam tie for the
   highest ROUGE-L (0.077)" 문장은 실제 결과에 맞게 다시 쓴다 (진짜 tie면 유지,
   아니면 어느 쪽이 이겼는지로 수정).
2. Table 11 필요시 교체 (Task B 결과에 따라).
3. §6.2 Error Analysis의 두 사례를 Task C 결과에 맞게 유지/교체.
4. §4.3 또는 Reproducibility에 캐시 버그 수정 사실을 Task D 문구로 추가.
5. Table 10/11이 바뀌면 그 수치를 인용하는 다른 문단(Introduction/Abstract에
   HotpotQA 관련 수치가 있다면, Conclusion 등)도 함께 검색해서 일치시킨다:
   ```bash
   grep -rn "0.077\|multi-hop\|HotpotQA" 로 docx 텍스트 전체를 훑어 확인
   ```

---

## Task F — Table 13 / Figure 9 (Statistical Robustness)가 Task A-E에서 누락됨

**확인된 문제:** `scripts/robust_stats_fair.py`의 `_best_hotpotqa_report()`는
`exp2_multihop_hotpotqa_*.json`을 glob으로 스캔해 "가장 큰 n"을 자동 선택하는데:

1. `_CONTAMINATED` 접미사를 붙여도 이 glob 패턴에 여전히 매치된다 (파일명이
   `exp2_multihop_hotpotqa_`로 시작하고 `.json`으로 끝나기만 하면 매치).
2. 오염된 `exp2_multihop_hotpotqa_20260630_021448.json`(n=100)과 Task A의 새 클린
   `exp2_multihop_hotpotqa_20260707_022936.json`(n=100)이 n에서 동점이라, 비교가
   `n > best_n`(엄격한 부등호)이라서 먼저 스캔되는 쪽이 이긴다 — glob 순서는
   보장되지 않으므로 **어느 쪽이 선택됐는지 알 수 없다.**
3. 실측: 문서의 현재 Table 13 HotpotQA 행(`vs BM25 p=0.020` 유의,
   `vs RAPTOR p=0.060` 비유의)은 Task A로 새로 고친 Table 10의 결론
   (`vs RAPTOR만 유의, p=0.018`)과 **정반대**다 → 오염된 파일이 여전히 쓰이고
   있다는 확정적 증거. 이 Table 13 수치를 그대로 시각화한 **Figure 9(CI forest
   plot)도 16행 중 8행(HotpotQA ROUGE-L ×4 + LLM-Judge ×4)이 오염된 값이다.**

부수적으로, 같은 스크립트의 `_best_general_report()`는 general 벤치마크로
`"online_local_llama_general_v3_n100.json"`을 하드코딩 우선순위로 쓰는데, Table 8과
Figure 2-4는 실제로 `v4_n100.json`을 쓴다 (둘 다 오염은 아니지만 서로 다른 실행분).
급한 문제는 아니니 여유 있으면 같이 정리, 없으면 건너뛰어도 됨.

**할 일:**

1. `_best_hotpotqa_report()` 수정: `_CONTAMINATED` 파일을 명시적으로 제외하고,
   동점일 때 "먼저 발견된 것"이 아니라 **파일명의 타임스탬프가 가장 최근인 것**을
   고르도록 변경 (예: 파일명에서 timestamp를 파싱해 비교, 또는 최소한
   `path.stat().st_mtime`로 최신 파일 우선).
   ```bash
   grep -n "_CONTAMINATED\|glob(" scripts/robust_stats_fair.py
   ```
2. (여유 있으면) `_best_general_report()`도 `v4_n100.json`을 `v3_n100.json`보다
   우선하도록 순서 조정 — Table 8/Figure 2-4와 근거 파일을 통일.
3. 재실행:
   ```bash
   python scripts/robust_stats_fair.py
   ```
   출력이 어느 파일(`_meta.file`)을 실제로 썼는지 로그로 확인.
4. **보고:** 새로 생성된 `robust_stats_fair_summary.json`의 HotpotQA
   ROUGE-L/LLM-Judge 부분 전체 (vs BM25/Dense/FlatRAG/RAPTOR 각각의
   Δ, [95% CI], d_z, p_perm, power, n₈₀) — 이걸로 제가 Table 13과 Figure 9를
   갱신합니다. general 쪽도 파일을 바꿨다면 그 결과도 함께 공유.

---

## 완료 후 보고 형식

- Task A: 새 파일 경로, 재검증 exact-match %, 새 Table 10 전체 수치, 기존 대비
  변화 요약.
- Task B: 새 category-breakdown 표, 기존 Table 11과의 차이.
- Task C: 두 사례가 클린 데이터에서 유지되는지/대체됐는지.
- Task D: 실제로 넣은 문장.
- Task E: 어떤 표/문단을 바꿨는지 diff 요약.
- Task F: `_best_hotpotqa_report()`/`_best_general_report()` 수정 diff, 재실행
  로그가 가리킨 실제 소스 파일, 새 HotpotQA(및 general) robust-stats 수치.

결과 주시면 그걸로 §5.9/§6.2/§4.3 문구, Table 13, Figure 9까지 정리하는 작업을
이어서 진행합니다.
