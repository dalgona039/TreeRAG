# PageTree-RAG — Adaptive Traversal Policy 실구현 + 검증 실행 프롬프트 (Claude Code)

> 목적: 논문 §3.3.1의 "Decision Rule" 문단(τ(q, depth), action ∈ {expand, prune,
> switch-strategy}, DFS/Beam 자동 선택 기준)이 **실제 코드에 존재하지 않는 기능을
> 서술한 것**임이 fact-check로 확인되었다. 이 프롬프트는 (1) 그 문단이 주장하는
> 메커니즘을 실제로 구현하고, (2) LLM 없이 돌릴 수 있는 offline 실험으로 검증하고,
> (3) (Ollama 있으면) fair generative protocol로 추가 검증한 뒤, (4) 논문 문구를
> 실측치로 다시 쓰는 것까지를 목표로 한다.

## 배경 — 무엇이 왜 틀렸는가

TreeRAG_TIST_ACM.docx §3.3.1에 다음 문단이 있다:

> "Formally, the adaptive traversal policy treats retrieval as a sequential
> decision problem... at each visited node v, the Decision Rule selects an
> action a in {expand, prune, switch-strategy} by comparing P(v | q) to an
> adaptive threshold tau(q, depth), and the traversal-selection criterion
> chooses DFS when the query's top-1 node score dominates its runner-up by a
> wide margin... and Beam Search when the score mass is spread over several
> candidate subtrees."

**실측 결과 (fact-check 완료, 아래 근거 그대로 인용 가능):**

1. **DFS/Beam 선택은 쿼리 적응형이 아니다.** `src/core/reasoner.py:165`에서
   `traversal_algorithm: TraversalAlgorithm = "beam_search"`는 생성자의 고정
   파라미터(기본값)다. 디스패치는 `src/core/reasoner.py:515`의 단순
   `if self.traversal_algorithm == "beam_search":` 뿐이며 쿼리로부터 계산되는
   곳이 전혀 없다. 게다가 메인 API 라우트 `src/api/routes.py:372-375`는
   `TreeRAGReasoner(selected_indices, use_deep_traversal=use_traversal)`처럼
   `traversal_algorithm` 자체를 전달하지 않아 항상 클래스 기본값(`beam_search`)이
   쓰인다. 요청 스키마(`src/api/models.py`)에도 모드 선택 필드가 없다.
2. **threshold는 q/depth에 안 변하는 고정 상수다.**
   - DFS: `src/core/tree_traversal.py:207-209`가 호출하는
     `ErrorRecoveryFilter.dual_stage_filter`는 `src/core/error_recovery.py:87`에서
     `is_relevant = combined_score > 0.5`로 하드코딩.
   - Beam: `src/core/reasoner.py:521`에서 `min_score_threshold=0.2`로 호출
     (참고: `beam_search.py:73`의 함수 시그니처 기본값은 0.3이지만 실제 호출부는
     0.2를 명시적으로 넘김).
   - `error_recovery.py:212-235`에 `adaptive_threshold_adjustment(num_selected,
     num_total, query_length)`라는, filter_rate와 query_length에 따라 threshold를
     실제로 바꾸는 메서드가 이미 존재한다. **그런데 `src/` 전체에서 이 메서드를
     호출하는 곳이 단 한 곳도 없다 (dead code).**
3. **"switch-strategy" (탐색 도중 알고리즘 전환)는 어디에도 없다.** 알고리즘은
   쿼리당 한 번 정해져 그 쿼리의 전체 탐색에 고정된다.

즉 이 문단은 "그럴듯한 정책 추상화"이지 "구현된 메커니즘의 서술"이 아니다.
리뷰어가 코드/재현성을 확인하면 바로 드러날 문제이므로, 논문에 남겨두는 것보다
**실제로 구현해서 진짜로 만드는 것**이 낫다는 판단으로 이 프롬프트를 작성한다.

## 절대 규칙

1. 수치 날조 금지. 아래 실험을 실제로 돌려서 나온 JSON에서만 논문 수치를 인용한다.
   실행이 안 되면 "안 됨"이라고 보고하고 해당 문장은 논문에 넣지 않는다.
2. 기존 `treerag_dfs` / `treerag_beam` 벤치마크 시스템과 그 결과(Table 3, 5, 7, 8,
   9, 10, 13 등 기존 논문 수치)는 **절대 건드리지 않는다.** 새 정책은 별도 시스템
   `treerag_auto`로 추가해서 나란히 비교한다 (기존 수치 무효화 방지).
3. docx 편집 전 반드시 백업 (`cp TreeRAG_TIST_ACM.docx
   TreeRAG_TIST_ACM_backup_$(date +%Y%m%d_%H%M%S).docx`), Word는 닫고 진행.
4. seed=42 고정, 재현 가능하게 스크립트 커밋.
5. margin cutoff 등 새로 도입하는 하이퍼파라미터는 반드시 "왜 이 값인가"를
   설명할 수 있어야 한다 (예: Table 6과 같은 방식으로 sensitivity sweep 하거나,
   최소한 2~3개 값을 시도해서 안정성을 보일 것). 하나의 값을 임의로 고정하고
   그걸로 최종 수치를 내지 말 것 — 이게 바로 지금 문제가 된 "heuristic"
   비판을 반복하는 것이다.

---

## Task 1 — τ(q, depth): adaptive threshold 실제 배선

**목표:** 이미 존재하지만 호출되지 않는 `adaptive_threshold_adjustment`를 depth까지
반영하도록 확장하고, DFS/Beam 양쪽의 실제 threshold 비교 지점에 연결한다.

1. `src/core/error_recovery.py:212`의 `adaptive_threshold_adjustment`에 `depth`
   파라미터를 추가한다. 현재는 `filter_rate`(선택률)와 `query_length`만 본다 —
   여기에 depth 항을 추가: 얕은 depth(1-2)에서는 재현율을 위해 threshold를 낮추고,
   깊은 depth(4+)에서는 정밀도를 위해 threshold를 높이는 방향
   (paper §3.3.1의 "structural(depth)" 항과 같은 방향성으로 서술 일관성 유지).
   예시 골격(값은 sweep으로 확정, 아래 참고):
   ```python
   def adaptive_threshold_adjustment(
       self, num_selected: int, num_total: int, query_length: int, depth: int = 1,
   ) -> float:
       base_threshold = self.confidence_threshold
       filter_rate = 1.0 - (num_selected / num_total) if num_total > 0 else 0.0
       depth_adj = -0.05 if depth <= 1 else (0.05 if depth >= 4 else 0.0)
       ...  # 기존 filter_rate / query_length 로직 유지, depth_adj를 더해 반환
       return base_threshold + depth_adj  # (기존 분기들과 결합)
   ```
2. `src/core/tree_traversal.py:207` 호출부 직전에서 이 함수를 실제로 호출해
   `dual_stage_filter`에 동적 threshold를 넘기도록 `dual_stage_filter`
   (`error_recovery.py:33`)에 `threshold: float = 0.5` 파라미터를 추가하고
   내부의 `is_relevant = combined_score > 0.5` (line 87)를
   `combined_score > threshold`로 바꾼다.
3. Beam Search도 동일 원칙: `src/core/reasoner.py:521`의 고정
   `min_score_threshold=0.2`를 depth별 sweep 단계(BeamSearchNavigator 내부에서
   depth 루프마다 갱신)로 바꾸거나, 최소한 depth=0 진입 시 한 번 계산해서 넘긴다.
4. **필수 검증:** 실제로 값이 바뀌는지 로그로 확인한다.
   ```bash
   grep -rn "adaptive_threshold_adjustment(" src/ benchmarks/
   # tree_traversal.py와 reasoner.py(혹은 beam_search.py) 양쪽에서 호출되어야 함
   ```

## Task 2 — 쿼리 적응형 DFS/Beam 자동 선택 (진짜 traversal-selection criterion)

**목표:** 논문이 주장하는 "top-1이 top-2를 크게 앞서면 DFS, 점수가 퍼져 있으면
Beam" 기준을 실제로 구현한다.

1. 새 함수 추가 (`src/core/adaptive_policy.py` 신규 파일 권장):
   ```python
   def choose_traversal_algorithm(
       root_children_scores: list[float], margin_cutoff: float = 0.15,
   ) -> str:
       """Return 'dfs' when the top candidate dominates the runner-up by
       >= margin_cutoff (low ambiguity -> precision-favoring exhaustive
       search); otherwise 'beam_search' (score mass spread across several
       subtrees -> coverage-favoring pruned search)."""
       if len(root_children_scores) < 2:
           return "dfs"
       ranked = sorted(root_children_scores, reverse=True)
       margin = ranked[0] - ranked[1]
       return "dfs" if margin >= margin_cutoff else "beam_search"
   ```
2. `TreeRAGReasoner.__init__` (`reasoner.py:161`)의 `traversal_algorithm` 타입에
   `"auto"`를 추가: `TraversalAlgorithm = Literal["dfs", "beam_search", "auto"]`.
   기본값은 하위 호환을 위해 그대로 `"beam_search"`로 둔다 (기존 실험 재현성 보존).
3. `_build_context_with_traversal` (`reasoner.py:506`)에서 `traversal_algorithm
   == "auto"`일 때: 루트의 depth-1 자식들에 대해 기존 semantic/keyword 스코어
   함수로 점수를 매긴 뒤 `choose_traversal_algorithm`을 호출해 실제 알고리즘을
   결정하고, 그 결과를 `trav_stats`에 `"auto_selected_algorithm"` 필드로 기록한다
   (나중에 논문에 "몇 %가 DFS로 갔는지" 보고하려면 이 로그가 필요).
4. **margin_cutoff 확정 방법 (절대 규칙 5 참고):** 0.10 / 0.15 / 0.20 세 값으로
   Task 3의 offline 실험을 각각 돌려 ROUGE-L과 DFS 선택 비율이 얼마나 민감한지
   보고 확정한다 (Table 6 스타일의 작은 sweep, 이 결과 자체도 논문에 넣을 수 있음).

## Task 3 — Offline 검증 (LLM 불필요, 지금 바로 실행 가능)

`treerag_auto`를 벤치마크 하네스에 추가한다.

1. `benchmarks/run_real_evaluation.py`:
   - `ALL_SYSTEMS`에 `"treerag_auto"` 추가, `SYSTEM_LABELS`에
     `"treerag_auto": "PageTree-RAG (Auto)"` 추가.
   - `run_system`에 분기 추가: `if system == "treerag_auto": return
     self._run_treerag(q, doc_id, "auto", branches)`.
   - `_run_treerag`의 오프라인 분기(약 L306-311, `keyword_traversal` 사용부)에
     `algo == "auto"`일 때 Task 2의 `choose_traversal_algorithm`으로 실제 분기해
     `keyword_traversal(..., prefer_shallow=...)`에 반영한다 (오프라인 모드는
     Ollama 없이 TF-IDF만 쓰므로 지금 바로 실행 가능).
2. 실행 (오프라인, 기존 `treerag_dfs`/`treerag_beam`과 나란히):
   ```bash
   python benchmarks/run_real_evaluation.py \
     --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam,treerag_auto \
     --dataset benchmarks/datasets/full_benchmark.json \
     --output data/benchmark_reports/offline_auto_policy_general_n204.json
   python benchmarks/run_real_evaluation.py \
     --systems treerag_dfs,treerag_beam,treerag_auto \
     --dataset benchmarks/datasets/medical_benchmark.json \
     --output data/benchmark_reports/offline_auto_policy_medical_n42.json
   ```
3. **보고 항목 (논문에 쓸 실측치):**
   - `treerag_auto`의 ROUGE-L / context tokens / latency vs. 기존
     `treerag_dfs`, `treerag_beam` 고정값.
   - 자동 선택에서 DFS를 고른 쿼리 비율 (%) — `trav_stats["auto_selected_algorithm"]`
     로그 집계.
   - margin_cutoff 0.10/0.15/0.20 세 값에서 위 수치가 얼마나 바뀌는지 (표 하나).
   - 만약 `treerag_auto`가 `treerag_dfs`/`treerag_beam` 중 더 나은 쪽에 근접하거나
     그것을 능가하면 이게 핵심 결과다. 못 미치면 **그것도 정직하게 보고** —
     "자동 선택이 고정 선택보다 못하다"도 논문에 쓸 수 있는 정직한 결과다
     (Limitations에 반영).

## Task 4 — (선택, Ollama 있는 환경에서만) Fair generative protocol 검증

이 세션(Cowork 샌드박스)에는 Ollama/GPU가 없어 여기서는 실행 불가. Ollama +
llama3.1:8b가 있는 로컬 환경에서 아래를 실행:

```bash
python benchmarks/run_real_evaluation.py \
  --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam,treerag_auto \
  --dataset benchmarks/datasets/full_benchmark.json \
  --limit 50 --seed 42 --gen-backend ollama --gen-model llama3.1:8b \
  --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
  --output data/benchmark_reports/online_local_llama_auto_policy_n50.json
```

**보고:** `treerag_auto`의 LLM-Judge / citation F1 / context tokens vs.
`treerag_dfs`(현재 논문의 fewest-token 시스템), `treerag_beam`(현재 논문의
best-judge 시스템). 핵심 질문: **자동 정책이 "둘 중 매번 다른 게 이기는" 상황에서
사람이 고르지 않아도 좋은 쪽에 가깝게 가는가?** 이게 Yes면 논문의 핵심 주장
("policy, not just a switch")이 실측으로 뒷받침된다.

## Task 5 — 논문(docx) 갱신

Task 3(+4)의 실측치가 나온 뒤 진행. Word 닫고, 백업 먼저.

1. §3.3.1 "Formally, the adaptive traversal policy..." 문단을 실측 기반으로
   재작성. 지금의 τ(q,depth)/switch-strategy 표현 대신:
   - 실제 depth-adjustment 공식 (Task 1에서 확정된 부호/크기)을 정확히 서술.
   - margin_cutoff와 그 sensitivity sweep 결과를 인용 (Table 6 옆에 작은 표나
     각주로 추가).
   - "switch-strategy" 표현은 삭제하거나, "알고리즘은 쿼리 시작 시 한 번
     결정되며 탐색 도중 전환하지 않는다"로 정정.
2. 새 표 추가 (예: Table 6b 또는 Table 14): `treerag_auto` vs 고정 DFS/Beam,
   offline (+ 있으면 fair protocol) 결과.
3. §6 Limitations에서 "adaptive threshold/policy selection is not yet learned
   or validated at scale"류 문장이 있다면, 이번에 실제로 구현/검증했다는 사실에
   맞게 갱신 (완전히 해결된 게 아니라면 "실구현했으나 margin_cutoff는 소규모
   sweep으로만 확정" 정도로 정직하게 남길 것).
4. Figure 9와 동일한 스타일(직렬 폰트, 옅은 grid, colorblind-safe palette)로
   Task 3/4 결과를 시각화하는 그림을 추가하고 싶으면
   `scripts/plot_results.py`의 `# UNIFIED_STYLE_INJECTED` rcParams 블록을
   그대로 재사용할 것 (이미 이번 라운드에서 적용됨).

---

## 완료 후 보고 형식

각 Task 종료 시 다음을 남길 것:
- Task 1-2: 어느 파일의 몇 번째 줄을 바꿨는지 diff 요약.
- Task 3(+4): 산출 JSON 경로, `treerag_auto` vs `treerag_dfs`/`treerag_beam`
  델타 표, margin_cutoff sweep 결과.
- Task 5: 어떤 문단/표를 어떻게 바꿨는지 (기존 논문 수치는 변경 없음을 재확인).

이 실측치들을 알려주면, 그걸 바탕으로 논문 문구를 다듬는 작업을 이어서 진행합니다.
