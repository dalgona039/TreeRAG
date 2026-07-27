# B1 — End-to-End Token/Call Accounting (the most important gap)

## Method

Added opt-in instrumentation (`src/utils/llm_accounting.py`, `TREERAG_TOKEN_ACCOUNTING=1`,
verified a true no-op when unset) at every LLM call site: `src/config.py` (Gemini path),
`src/core/ollama_client.py` (Ollama path — the backend actually used here),
`benchmarks/metrics/llm_judge.py`'s `LocalJudge` (a third code path that hits Ollama's
HTTP API directly), and `src/core/raptor_ollama_adapter.py` (RAPTOR's indexing-time
QA/summarization calls). Each record captures pipeline stage, backend, model, real
prompt/completion token counts (from Ollama's own `prompt_eval_count`/`eval_count`, or
Gemini's `usage_metadata`), and latency.

Ran the **fair protocol** (Full Benchmark, n=100, seed=42, same setting as Table 8) with
accounting enabled, all 6 systems, real fixed baselines (Dense/BERTScore/RAPTOR — see
B3). Report: `data/benchmark_reports/online_local_llama_general_b1_20260727_n100.json`.
Accounting dump: `..._n100_token_accounting.json` (1991 call records).

**A real bug was found and fixed mid-run**: Dense Retrieval returned empty answers for
all 100 questions (a stale 256-dim cache from the old hash-embedding fallback collided
with the new 768-dim real embeddings — fixed in `dense_retrieval_baseline.py`, see B3's
update log). Dense was re-run standalone after the fix
(`..._dense_fixed_n100.json` / `..._dense_fixed_n100_token_accounting.json`, 200 records)
and those corrected numbers are used below.

## ★ Headline finding: PageTree-RAG makes far more LLM calls per query than any baseline

| System | LLM calls/query | Prompt tok/query | Completion tok/query | Total tok/query | Generator ctx tok/query | Latency (s/query) |
|---|---|---|---|---|---|---|
| BM25 | 2.0 | 608.8 | 116.9 | 725.7 | 113.4 | 6.62 |
| Dense Retrieval | 2.0 | 591.0 | 111.4 | 702.3 | 110.7 | 7.87 |
| FlatRAG | 2.0 | 606.9 | 111.5 | 718.4 | 0‡ | 6.21 |
| RAPTOR (query-time) | not reliably isolated (§ Reconciliation gap) | — | — | — | 214.7 | 60.10 |
| PageTree-RAG (DFS) | 5.76 | 1885.6 | 417.4 | 2302.9 | 7.45 | 41.00 |
| PageTree-RAG (Beam) | 3.97 | 2408.9 | 742.5 | 3151.4 | 18.67 | 78.74 |

‡ FlatRAG's 0 generator-context-token instrumentation gap is pre-existing and already
documented in the paper (Table 8's own footnote); unrelated to this accounting pass.

Baselines make exactly 2 calls/query (1 generation + 1 LLM-judge call, both part of the
*evaluation harness*, not the retrieval system itself). **PageTree-RAG (Beam) makes
~2× and DFS ~2.9× as many calls**, because tree traversal itself requires one or more
LLM node-scoring calls before generation ever happens (this is exactly what Table 8's
"Generator ctx tokens" column — the paper's current sole efficiency metric — does not
capture, since it only counts what reaches the final generation call).

**PageTree-RAG (Beam) uses ~4.3× the total tokens per query of BM25/Dense/FlatRAG**
(3151 vs. ~715 average), despite delivering **6–15× fewer tokens to the generator itself**
(18.7 vs. 110–113). The paper's "token-efficient" framing is accurate about the
generation-time context window, and the title/abstract should be read that way
specifically — but it is not true of end-to-end token cost, where PageTree-RAG is more
expensive, not less. **This directly confirms the concern that motivated B1**: the
current paper counts only generator-context tokens and never discloses call count or
end-to-end token totals.

## RAPTOR: separately measured indexing overhead (real, one-time)

RAPTOR requires real tree-building (GMM clustering + LLM summarization, see B3) the
first time each document is seen, cached thereafter per document, not per query.

| | Value |
|---|---|
| Unique documents in the n=100 sample | 25 |
| Total indexing LLM calls (all 25 docs) | 318 (12.7 calls/doc average) |
| Total indexing tokens (prompt+completion) | 181,656 |
| Total indexing wall-clock | 5323s ≈ 1.48 hours (one-time, amortized across however many future queries touch these 25 documents) |

This cost does not appear anywhere in the paper. Even amortized across all 100 queries
in this sample, it adds ~1817 tokens/query and ~53s/query of one-time-but-real cost that
the current token/latency tables do not account for at all.

## ★ Reconciliation gap: RAPTOR's query-time generation/judge calls

RAPTOR's per-question log shows only 3/100 questions fell back to extractive answers
(`⚠️ LLM returned very short answer`), meaning ~97 real LLM generation calls succeeded —
but the accounting stream's `generation_baseline` stage total (300 records) is fully and
exactly accounted for by BM25+Dense+FlatRAG (100 each), leaving **zero** attributable to
RAPTOR. This is an unresolved instrumentation gap, not a claim that RAPTOR made no
calls — the calls demonstrably happened (per_question latency for RAPTOR, 60.1s/query,
is real and far exceeds BM25/Dense/FlatRAG's 6–8s, consistent with real LLM calls plus
amortized indexing) — but this pass could not cleanly attribute their token counts to
RAPTOR specifically rather than to the immediately-following DFS system in the flat call
log. Root cause not found in the time available; the honest fix is adding a
system-name/query-index field to each accounting record (not just pipeline stage),
which was not done since it would require re-running the full 9-hour benchmark again.
**Flagged as unmeasured** — do not infer RAPTOR made 0 query-time calls; only that this
pass cannot state its exact token count.

The DFS/Beam split itself is reliable (verified: Beam's block is a clean
198×traversal_beam + 100×judge + 99×generation with no stray stages, and 100% of DFS's
extra "unknown"-stage calls were traced to a second, previously-uninstrumented
traversal call site in `tree_traversal.py:181` — `llm_evaluate`, a per-node relevance
judgment used by DFS's traversal/error-recovery path, now understood but not
retroactively re-tagged for this already-completed run).

## Cost estimate (Gemini 2.5 Flash pricing, for reference)

Actual backend used throughout is local Ollama (llama3.1:8b) — **zero API cost**. For a
hosted-API cost estimate, using Gemini 2.5 Flash's published rate — **$0.30 / 1M input
tokens, $2.50 / 1M output tokens** (Google AI pricing, per multiple July 2026 pricing
trackers; Gemini 2.5 Flash is slated for deprecation 2026-10-16, successor models are
priced higher) — applied to the real measured token counts above:

| System | Est. cost / 100 queries | Est. cost / query |
|---|---|---|
| BM25 | $0.0475 | $0.00048 |
| Dense Retrieval | $0.0456 | $0.00046 |
| FlatRAG | $0.0461 | $0.00046 |
| PageTree-RAG (DFS) | $0.1609 | $0.00161 |
| PageTree-RAG (Beam) | $0.2579 | $0.00258 |
| RAPTOR indexing (one-time, 25 docs) | $0.1494 | — |
| RAPTOR query-time | not measured (see reconciliation gap) | — |

**At current Flash pricing the absolute dollar amounts are trivially small either way**
(fractions of a cent per query for every system) — cost in dollars is not where the
"token-efficient" framing is actually at risk. The meaningful costs are **call count**
(rate limits, quota, engineering complexity of a multi-call pipeline vs. a single call)
and **latency** (Beam at 78.7s/query vs. BM25 at 6.6s/query is a real, user-facing
12× difference that the paper already reports in Table 8's latency column, but without
explaining *why* — this accounting is the "why": it's driven by call count, not by
context size).

## Recommended paper framing change

Add a table immediately after Table 8 (suggested "Table 8b: End-to-end token/call
accounting") using the headline table above, and revise the "token-efficient" claims in
the abstract/title to specify **generation-context-token-efficient**, not end-to-end
token- or cost-efficient. Suggested one-sentence addition to the current framing:

> *PageTree-RAG trades generation-time context tokens for retrieval-time LLM calls: it
> delivers 6–15× fewer tokens to the final generation step than any baseline (Table 8),
> but at 2–3× the total LLM call count and total token volume per query, concentrated in
> the traversal stage rather than generation (Table 8b).*

## Answered vs. unmeasured

- **Answered**: real LLM-calls-per-query, prompt/completion/total tokens per query, and
  latency per query for BM25, Dense (corrected), FlatRAG, PageTree-RAG DFS, and
  PageTree-RAG Beam. Real one-time RAPTOR indexing cost (calls, tokens, wall-clock)
  across all 25 unique documents in the sample. A real, sourced hosted-API cost estimate
  at current Gemini 2.5 Flash pricing.
- **Not measured**: RAPTOR's query-time generation/judge call count and token totals
  could not be cleanly separated from DFS's in this run's flat accounting log (see
  Reconciliation gap above) — its latency (60.1s/query) and generator-context tokens
  (214.7/query) are separately reliable (from the evaluation framework's own per-question
  tracking, unaffected by the accounting-stream ambiguity). A rerun with a
  system-name-tagged accounting record would resolve this cleanly but was not performed
  in this session.
