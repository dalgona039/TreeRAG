# B3 — Baseline Hyperparameters (code audit, no experiments run)

Method: static read of `src/core/*.py` and the actual driver used to produce the
stored result files in `data/benchmark_reports/` (`benchmarks/run_real_evaluation.py`).
No numbers in this document are estimated.

**Important scope note discovered during this audit**: there are *two* independent
FlatRAG-shaped implementations in this repo:

1. `src/core/flat_rag_baseline.py` — `FlatRAGBaseline`, imported and used by
   `benchmarks/run_real_evaluation.py::_run_flatrag` (the script that produced the
   `data/benchmark_reports/online_local_llama_*` and `offline_auto_policy_*` files,
   i.e. the numbers actually reported in the paper).
2. `benchmarks/compare_baselines.py` — a separate, self-contained chunker/retriever
   (`chunk_size: int = 512` characters at line 132) that appears to be an older/parallel
   harness. It is **not** what produced the stored result files (no `data/benchmark_reports/*`
   file references `compare_baselines.py` in its content, and `run_real_evaluation.py`
   never imports it).

The table below reports values from path (1), since that is what generated the paper's
numbers, and flags (2) separately so it doesn't get confused with the real pipeline.

| 항목 | 값 | 출처 (파일:줄) |
|---|---|---|
| 청크 크기 — RAPTOR | 300 chars (`CHUNK_SIZE`) | `src/core/raptor_baseline.py:23,58` |
| 청크 크기 — FlatRAG(actual, used) | **코드에서 확인 불가** — `FlatRAGBaseline` in `src/core/flat_rag_baseline.py` does not chunk by fixed size in the reviewed sections; it retrieves over existing PageIndex tree nodes (title+summary), not fresh fixed-size chunks. See caveat below. | `src/core/flat_rag_baseline.py` (class def ~L85–L260) |
| 청크 크기 — `compare_baselines.py`'s own chunker (NOT used for paper numbers) | 512 characters | `benchmarks/compare_baselines.py:132` |
| 청크 크기 — PageTree-RAG tree nodes | Nodes come from the PageIndex tree structure (`src/core/indexer.py`), not fixed-size chunking; granularity = document section, not token count | `src/core/indexer.py` |
| 청크 overlap | **코드에서 확인 불가** — no `overlap` parameter found in any of the reviewed retrieval/chunking code paths | — |
| top-k — BM25 | passed as `branches` from the eval driver; CLI default `branches=3` per `run_system(..., branches: int = 3)` default | `src/core/bm25_baseline.py:64` (`search(query, top_k)`); `benchmarks/run_real_evaluation.py:204-209,335` |
| top-k — Dense | same `branches` value as BM25 (shared driver param) | `src/core/dense_retrieval_baseline.py:172`; `benchmarks/run_real_evaluation.py:217-222` |
| top-k — FlatRAG | `max_branches=branches` (same shared value) | `src/core/flat_rag_baseline.py` `.query(q, max_branches=branches)`; `benchmarks/run_real_evaluation.py:230-237` |
| top-k — RAPTOR | `max(branches, 5)` — i.e. floor of 5 regardless of the shared `branches` value | `benchmarks/run_real_evaluation.py:275-282` |
| BM25 k1, b | k1 = 1.5, b = 0.75 (library defaults, not overridden anywhere found) | `src/core/bm25_baseline.py:12` |
| Dense: Sentence-BERT 체크포인트명 | Attempted in order: `jhgan/ko-sroberta-multitask` (`KO_MODEL`) → `intfloat/multilingual-e5-base` (`FALLBACK_MODEL`) → **deterministic hashing embedder** (`HashingEmbedder`, 256-dim) if `sentence-transformers` import or both model loads fail | `src/core/dense_retrieval_baseline.py:37-38,78-93` |
| Dense: 유사도 함수, 정규화 여부 | Cosine similarity via L2-normalized vectors (`_normalize`) then inner product; FAISS `IndexFlatIP` if `faiss` installed, else NumPy inner-product scan | `src/core/dense_retrieval_baseline.py:53,145,172-177` |
| **Dense: which path actually ran** | `sentence-transformers`, `torch`, and `faiss` are **not** in `requirements.txt` and are **not installed** in `.venv` (checked directly: `pip list` shows neither package). The `jhgan/ko-sroberta-multitask` checkpoint is also Korean-only, mismatched for the English HotpotQA/GovReport benchmarks even if it did load. High-confidence conclusion: the "Dense" baseline results in the stored JSON files were produced by the **hashing fallback embedder**, not real sentence embeddings. This could not be verified retroactively per-run because the result JSONs don't log which embedder branch fired — flagged as a finding, not a measured fact for prior runs. | `requirements.txt` (no ML deps listed); `.venv/bin/pip list` (checked live) |
| FlatRAG: 60/25/15 가중치의 정의와 출처 | `0.6 * normalized_bm25 + 0.25 * semantic_score + 0.15 * structural_score`. No citation or derivation found in code or comments — appears to be an author-chosen constant. See B4 for the sensitivity-sweep follow-up. | `src/core/flat_rag_baseline.py:254-256` |
| RAPTOR: 클러스터 수/방법, 요약 모델, 트리 높이 | **`raptor` pip package is not installed** (`import raptor` fails; not in `requirements.txt`). All RAPTOR results were produced by `RaptorFallback`, a deterministic, non-LLM, non-semantic placeholder: clusters = round-robin grouping of `CLUSTER_SIZE=3` chunks (not GMM/k-means as in the real RAPTOR paper), "summary" = literal concatenation of each member's first `SUMMARY_PREFIX_CHARS=80` characters (no LLM call), tree height = fixed 2 levels (chunk → cluster), retrieval score = character-trigram Jaccard overlap (no embeddings). **This means the paper's RAPTOR baseline is not RAPTOR** — it is a crude extractive proxy that happens to share the name. This is a significant baseline-validity issue beyond the original B1–B7 list; recommend treating it with the same urgency as B4's FlatRAG-weights issue. | `src/core/raptor_baseline.py:23-24,79-91 (CLUSTER_SIZE, SUMMARY_PREFIX_CHARS),95-150 (RaptorFallback),176-177 (import raptor try/except)` |
| PageTree-RAG: max_branches b, max_depth d, beam width W | Class defaults: `max_depth=5, max_branches=3` (DFS/`TreeNavigator`); `beam_width=5` (`BeamSearchNavigator`). Driver default: `branches=3` passed to `run_system`; `margin_cutoff=0.15` default for `auto` mode. **Actual per-run values used for each stored result file were not re-derived from CLI invocation logs** — the JSON result files do not consistently log the branches/depth/beam-width CLI args used (some do, e.g. `margin_cutoff` is echoed at `run_real_evaluation.py:740`; top-level `branches`/`max_depth` echoing was not found for all files). Flagged as **코드 기본값은 확인, 실행별 실제값은 미측정** for anything beyond `margin_cutoff`. | `src/core/tree_traversal.py:24-25`; `src/core/beam_search.py:53-61`; `benchmarks/run_real_evaluation.py:335,651,740` |

## ★ Chunk-size finding for the "10× fewer context tokens" claim

The paper's "10× fewer context tokens" claim needs a chunk-token denominator for the
**chunk-based baselines that actually produced the reported numbers** (BM25, Dense,
FlatRAG via `run_real_evaluation.py`). Of those:

- BM25/Dense/FlatRAG in `run_real_evaluation.py` do **not** re-chunk documents at
  query time — they retrieve over the **same pre-built PageIndex tree nodes** as
  PageTree-RAG (`self.load_tree(doc_id)` is shared across all systems; see
  `run_real_evaluation.py:147-154,204-243`). Chunk granularity for these baselines is
  therefore whatever granularity the tree-building/indexing step (`src/core/indexer.py`)
  produced, not a fixed token/character chunk size.
- Only RAPTOR does independent fixed-size chunking (`CHUNK_SIZE=300` chars,
  `src/core/raptor_baseline.py:23`), and only `compare_baselines.py`'s (unused for
  paper numbers) internal baseline uses `chunk_size=512` chars.

**Conclusion: 코드에서 명시적 "chunk size" 상수를 baseline과 PageTree-RAG가 공유하는 형태로는 확인 불가.**
The "10× fewer context tokens" comparison in the paper is not measuring chunk-size vs.
chunk-size; it's measuring context-tokens-passed-to-generator, which is a different and
separate quantity (see B1). Recommend the paper either (a) drop "chunk size" framing in
favor of "context tokens delivered to the generator" (already measured via
`contextual_compressor.py`'s token counting, `max_output_tokens=4000` cap at
`src/core/contextual_compressor.py:36`), or (b) explicitly state that baselines retrieve
over the same tree nodes as PageTree-RAG rather than independent fixed-size chunks.

## Update (2026-07-27): Dense/BERTScore/RAPTOR fallbacks fixed going forward

Everything above describes the state that produced the results currently in
`data/benchmark_reports/` and the paper. That same day, the missing
dependencies were installed and the fallbacks fixed for **future** runs:

- `sentence-transformers`, `bert-score`, `torch` installed and added to
  `requirements.txt`. Verified live: Dense now loads real
  `jhgan/ko-sroberta-multitask` embeddings (not `HashingEmbedder`); BERTScore
  now returns real contextual scores (0.97 on a close paraphrase, vs. the
  0.2–0.5 range the token-F1 proxy produced — see B5).
- RAPTOR: the official implementation (github.com/parthsarthi03/raptor, MIT)
  is vendored at `third_party/raptor` (not pip-installable under that name —
  the PyPI package `raptor` is an unrelated tool) and wired to a local Ollama
  QA/summarization adapter (`src/core/raptor_ollama_adapter.py`) instead of
  RAPTOR's default OpenAI backend, so no OpenAI key is needed. Verified live
  on a 50-paragraph synthetic document: real GMM clustering triggered and
  produced genuine LLM-generated abstractive cluster summaries via
  `llama3.1:8b` (e.g. *"The Amazon rainforest has been reduced by
  approximately 17% since 1970 due to deforestation..."*), not the old
  extractive first-80-characters stub. Build time for that 50-paragraph doc
  was 93s (4 real summarization calls) — real documents in the benchmark
  corpus will take substantially longer per document to index than the old
  fallback, which was near-instant. This has real implications for how long
  a full RAPTOR rerun over 100 questions takes and should be budgeted for
  before re-running B1/B7/B2/B4 against the fixed baselines.
- Two bugs were found and fixed while verifying this: (1) a thread-safety
  race in the embedder cache that made RAPTOR's concurrent leaf-node
  embedding calls each load their own copy of the model, causing an OOM-style
  silent death (fixed with a lock, `src/core/raptor_ollama_adapter.py`); (2) a
  torch MPS (Apple GPU) backend segfault (`EXC_BAD_ACCESS`/`SIGSEGV`, macOS
  crash report captured) during embedding — fixed by forcing
  `device="cpu"` in both the shared embedder (`dense_retrieval_baseline.py`)
  and `bertscore_f1()` (`benchmarks/metrics/text_similarity.py`).
- One correctness patch to the vendored RAPTOR code was needed: upstream's
  unguarded `n_neighbors = int(sqrt(n-1))` for UMAP is 0 or 1 for small leaf
  counts and hung indefinitely; clamped to >=2
  (`third_party/raptor/cluster_utils.py`, documented in `third_party/README.md`).

**Not yet done**: none of `data/benchmark_reports/`'s existing results have
been regenerated with the fixed baselines — that requires a full rerun
(B1/B7/B2/B4 territory) and is a separate, larger decision given the RAPTOR
timing finding above.

## Items not measured (for Limitations)

- FlatRAG's actual per-node "chunk size" in characters/tokens (depends on how
  `src/core/indexer.py` built the tree — not re-derived here since it requires reading
  the indexing pipeline in depth, out of scope for a hyperparameter audit).
- Chunk overlap — no such parameter exists anywhere in the reviewed code paths.
- Per-run actual `max_branches`/`max_depth`/`beam_width` CLI values for each specific
  stored result file (only code defaults were confirmed with certainty).
- Whether the Dense baseline's embedder fell back to hashing for the *specific* runs
  that produced the paper's numbers (inferred with high confidence from missing
  dependencies, not directly logged per-run).
