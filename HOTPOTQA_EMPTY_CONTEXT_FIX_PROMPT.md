# PageTree-RAG — HotpotQA 빈 컨텍스트 버그 수정 + 전면 재실행 프롬프트 (Claude Code)

> **이 프롬프트는 `MULTIHOP_CACHE_FIX_RERUN_PROMPT.md`의 Task A/C(HotpotQA n=100
> 재실행 및 케이스 스터디)를 대체한다.** 그 프롬프트가 고친 캐시 키 버그는 여전히
> 유효하고 필요하지만, 그보다 더 근본적인 버그가 있었다: n=100 실행이 참조한
> 인덱스 100개가 전부 **빈 컨텍스트(children=0)**였다. 캐시 오염을 없애고
> 재실행해도,애초에 검색 대상 문서가 없었으므로 결과는 여전히 무효다.

## 배경 — 무엇이, 왜, 얼마나 깨졌는가

`benchmarks/datasets/hotpotqa_loader.py`의 `_normalize_item()`이 공식 HotpotQA
dev set(`data/hotpotqa/hotpot_dev_distractor_v1.json`, 6/30에 로컬 추가된 진짜
official 파일)의 실제 스키마와 다르게 짜여 있었다.

**실제 공식 스키마** (`context`, `supporting_facts` 둘 다 병렬-리스트 딕셔너리):
```json
"context": {"title": ["Ed Wood (film)", "Scott Derrickson"],
            "sentences": [["Ed Wood is a 1994...", "..."], ["Scott Derrickson...", "..."]]}
"supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]}
```

**코드가 가정한 스키마** (list-of-pairs — 번들된 20문항 샘플 `hotpotqa_sample.json`은
이 형태라 안 깨졌음):
```python
"context": [["Ed Wood (film)", ["Ed Wood is a 1994...", "..."]], ...]
```

`for ctx in raw.get("context", []) or []:`가 dict를 순회하면 키 문자열
(`"title"`, `"sentences"`)만 나오고, 이건 `isinstance(ctx, (list,tuple))`도
`isinstance(ctx, dict)`도 통과 못 해서 **아무것도 append되지 않는다.** 결과:
`data/indices/hotpotqa_hp_0_index.json` ~ `hp_99_index.json` **100개 전부
`children: []`**. (번들 20문항 `hotpotqa_hp_sample_*_index.json`은 정상.)

덤으로 `raw.get("_id", "")`도 실제 키가 `"id"`라서 `question_id`가 항상 빈
문자열이었다 (이게 이전에 `question_id` 매칭이 실패해 positional index로 우회해야
했던 이유).

**피해 범위가 PageTree-RAG에 국한되지 않는다.** `benchmarks/run_real_evaluation.py`의
`_run_bm25`/`_run_dense`/`_run_flatrag`가 전부 `self.load_tree(doc_id)`로 **같은
빈 트리**를 읽는다 (`BM25Retriever(self.load_tree(doc_id))` 등). 즉 이 n=100
실행에서는 **6개 시스템 전부**가 빈 컨텍스트로 "검색"했고, 답변이 그럴듯했다면
그건 검색이 아니라 Llama 3.1 8B의 사전학습 지식(HotpotQA가 위키피디아 기반이라
학습 데이터에 있었을 가능성)에서 나온 것이다. `treerag_beam`이 `treerag_dfs`보다
실패(judge ≤ 0.4)가 많았던 것도, 검색 알고리즘 차이가 아니라 "모른다"고 정직하게
답한 경우와 사전지식으로 추측해 맞힌 경우의 차이였을 가능성이 있다.

**영향받는 것: Table 10, Figure 3(multi-hop 관련 부분)/Figure 4, Table 11(일반
벤치마크가 소스라 직접 영향은 없으나 §5.9 텍스트에서 같이 언급됨 — 재확인만),
Table 13/Figure 9의 HotpotQA 행, §5.9 전체, §6.2 케이스 스터디.** 영향받지
않는 것: general/medical/govreport 벤치마크 (완전히 다른 인덱싱 파이프라인 —
LLM 기반 `RegulatoryIndexer`로 만든 실제 PDF 인덱스이지 이 로더를 쓰지 않음).

## 절대 규칙

1. 수치 날조 금지. 재실행 결과에서만 인용.
2. 고치기 전에 `_normalize_item`이 정말 매핑을 실패하는지 **먼저 재현**한다
   (아래 Task 0). 재현 없이 바로 고치지 말 것 — 다른 원인일 가능성 배제.
3. 인덱스 재생성 전 기존 깨진 인덱스를 백업 (`data/indices_BROKEN_backup_<ts>/`).
4. 이전에 만든 오염 파일 보관 규칙 계속 유지 (`_CONTAMINATED` 접미사born 파일은
   그대로 두고 건드리지 않음).
5. seed=42 고정.

---

## Task 0 — 버그 재현 (수정 전 먼저 확인)

```bash
python3 - <<'PY'
import json
d = json.load(open("data/hotpotqa/hotpot_dev_distractor_v1.json"))
item = d[0]
print("context type:", type(item["context"]))
print("supporting_facts type:", type(item["supporting_facts"]))

import sys
sys.path.insert(0, ".")
from benchmarks.datasets.hotpotqa_loader import _normalize_item
norm = _normalize_item(item)
print("normalized context len:", len(norm["context"]))       # 지금은 0이어야 버그 재현됨
print("normalized supporting_facts len:", len(norm["supporting_facts"]))  # 지금은 0
print("question_id:", repr(norm["question_id"]))              # 지금은 '' 이어야 함
PY
```
`context len`이 0이 아니면 이미 다른 버전이거나 가정이 틀린 것이니, 먼저 그 차이를
보고할 것.

## Task 1 — `_normalize_item()` 수정

`benchmarks/datasets/hotpotqa_loader.py`의 `_normalize_item` (약 L46-67)을 아래로
교체 — **딕셔너리(공식 포맷)와 리스트(번들 샘플 포맷) 둘 다 지원**하도록 작성:

```python
def _normalize_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a native HotpotQA item to the spec's dict schema.

    Handles both the official HotpotQA dev-set schema (context/supporting_facts
    as a dict of parallel lists: {"title": [...], "sentences": [[...], ...]})
    and the list-of-pairs schema used by the bundled 20-question sample.
    """
    supporting = []
    sf = raw.get("supporting_facts")
    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        for t, sid in zip(titles, sent_ids):
            supporting.append({"title": t, "sent_id": sid})
    elif isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                supporting.append({"title": item[0], "sent_id": item[1]})
            elif isinstance(item, dict):
                supporting.append({"title": item.get("title", ""), "sent_id": item.get("sent_id", 0)})

    context = []
    ctx = raw.get("context")
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for t, sents in zip(titles, sentences):
            context.append({"title": t, "sentences": list(sents)})
    elif isinstance(ctx, list):
        for c in ctx:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                context.append({"title": c[0], "sentences": list(c[1])})
            elif isinstance(c, dict):
                context.append({"title": c.get("title", ""), "sentences": c.get("sentences", [])})

    return {
        "question_id": raw.get("_id") or raw.get("id", ""),
        "question": raw.get("question", ""),
        "answer": raw.get("answer", ""),
        "type": raw.get("type", "bridge"),
        "supporting_facts": supporting,
        "context": context,
    }
```

Task 0의 스크립트를 다시 돌려 `context len > 0`, `supporting_facts len > 0`,
`question_id`가 실제 official id 문자열인지 확인.

## Task 2 — 인덱스 재생성 + 검증

1. 기존(깨진) 인덱스 백업:
   ```bash
   mkdir -p data/indices_BROKEN_backup_$(date +%Y%m%d_%H%M%S)
   cp data/indices/hotpotqa_hp_[0-9]*_index.json data/indices_BROKEN_backup_*/
   ```
2. 재생성 (기존 파이프라인 재사용):
   ```bash
   python benchmarks/run_exp2_multihop.py --n 100 --seed 42 --smoke --limit 5
   ```
   (스모크 먼저 — Part A 앞부분에서 인덱스를 다시 쓰는지 확인. 인덱스 생성이
   별도 함수라면 `convert_to_benchmark_format(write_indices=True)` 직접 호출.)
3. **검증 (필수):**
   ```bash
   python3 - <<'PY'
   import json, glob
   bad = []
   for f in sorted(glob.glob("data/indices/hotpotqa_hp_*_index.json"))[:100]:
       d = json.load(open(f))
       n = len(d.get("children", []))
       if n == 0:
           bad.append(f)
   print(f"empty-children files: {len(bad)} / 100")
   if bad:
       print(bad[:5])
   # 샘플 하나 내용 눈으로 확인
   d = json.load(open("data/indices/hotpotqa_hp_0_index.json"))
   print(json.dumps(d, ensure_ascii=False, indent=2)[:600])
   PY
   ```
   `empty-children files`가 0이어야 하고, 샘플의 `children[0].summary`에 실제
   위키피디아 문장이 들어있어야 한다 (지금처럼 비어 있으면 안 됨).

## Task 3 — BM25/Dense/RAPTOR도 같은 문제였는지 확인

`_run_raptor`도 `self.load_tree(doc_id)`를 쓰는지 확인:
```bash
grep -n "load_tree\|def _run_raptor" -A 8 benchmarks/run_real_evaluation.py | grep -A8 "_run_raptor"
```
전부 같은 `load_tree`를 공유한다면(사실상 확정적), Task 2의 인덱스 재생성만으로
6개 시스템 모두 자동으로 고쳐진다 — 별도 조치 불필요. 결과만 보고.

## Task 4 — 전면 재실행 (캐시 버그 수정 + 빈 컨텍스트 수정 둘 다 반영된 상태)

```bash
python benchmarks/run_exp2_multihop.py --n 100 --seed 42 --skip-partB
```
직후 두 가지 검증:
```bash
# (a) 캐시 오염 재검증 (여전히 깨끗해야 함)
python scripts/check_cache_contamination.py data/benchmark_reports/exp2_multihop_hotpotqa_<new_ts>.json
# (b) 컨텍스트가 실제로 쓰였는지 — context_tokens가 0 근처가 아니어야 함
python3 -c "
import json
d = json.load(open('data/benchmark_reports/exp2_multihop_hotpotqa_<new_ts>.json'))
for s, recs in d['per_question'].items():
    toks = [r.get('context_tokens', 0) for r in recs]
    print(s, 'mean context_tokens=', sum(toks)/len(toks), 'min=', min(toks))
"
```

## Task 5 — (진단, 논문 투명성용) 이전 결과가 "암기"였는지 정량화

이전(빈 컨텍스트) 답변들 중 몇 %가 "정직한 모름"이었고 몇 %가 "그럴듯한 암기
추측"이었는지 세면 Discussion에 쓸 좋은 근거가 된다:

```bash
python3 - <<'PY'
import json, re
d = json.load(open("data/benchmark_reports/exp2_multihop_hotpotqa_20260707_022936.json"))  # 이전(빈 컨텍스트) 파일
refusal_pat = re.compile(r"no information|not (mentioned|contain|provided)|don'?t know|cannot determine|정보가 없|알 수 없", re.I)
for sysname in ["treerag_dfs", "treerag_beam", "bm25"]:
    recs = d["per_question"][sysname]
    refusals = sum(1 for r in recs if refusal_pat.search(r.get("answer","")))
    print(sysname, f"{refusals}/{len(recs)} refusal-like answers")
PY
```

## Task 6 — 논문(docx) 반영

Task 4 결과가 나온 뒤, `MULTIHOP_CACHE_FIX_RERUN_PROMPT.md`의 Task E/Task F가
다루던 위치(Table 10/11 텍스트, §5.9, §6.2 케이스 스터디, Table 13/Figure 9의
HotpotQA 행, Figure 3/4)를 **이번에 나온 진짜 클린 수치로** 다시 채운다. 이전
"클린"이라고 믿었던 07-06/07-07 실행분들도 이번 기준으로는 무효였음을 §6.2 또는
Reproducibility에 짧게 덧붙일 것 (언어-드리프트, 캐시-키 버그에 이은 세 번째
투명 공개 사례).

---

## 완료 후 보고 형식

- Task 0: 재현 확인 결과.
- Task 1: diff.
- Task 2: empty-children 개수(수정 전/후), 샘플 인덱스 내용.
- Task 3: RAPTOR도 같은 경로인지.
- Task 4: 새 리포트 파일 경로, 캐시 재검증 %, 평균 context_tokens.
- Task 5: refusal-rate 비교표.
- 새 Table 10/11/13, Figure 9(HotpotQA 행) 수치 — 이걸로 최종 반영하겠습니다.
