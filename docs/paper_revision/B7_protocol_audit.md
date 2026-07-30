# B7 — Offline (keyword-only) vs. Full-LLM Evaluation Protocol Audit

## 1. How to tell offline from online in the stored result files

Every `data/benchmark_reports/*.json` file produced by `benchmarks/run_real_evaluation.py`
carries a top-level `"mode"` field (`"offline"` or `"online"`), and offline-mode
`latency` values are always sub-millisecond to a few milliseconds (pure keyword
scoring, no network/LLM call), while online-mode latencies are seconds
(matches real LLM call round-trips). This makes the two trivially distinguishable
even where the field is missing, by inspecting `per_question[*].latency`.

`detect_mode()` (`benchmarks/run_real_evaluation.py:402-411`): with `--mode auto`
(the default), the runner pings the active LLM client once; on failure it silently
drops to the offline keyword+extractive fallback path (`Evaluator._run_treerag`'s
`# offline keyword approximation` branch, `run_real_evaluation.py:317-333`, using
`keyword_traversal()` / token-F1 node scoring instead of any LLM call). This is the
mechanism §4.5 refers to as the fallback for LLM API outages.

## 2. Table-by-table verdict

Each row below is either **byte-exact matched** (every reported number in the docx
table reproduces to 3 decimals from the named JSON file — same standard of proof used
in B5 for Table 8) or **caption-only** (the table's own caption states its protocol;
the underlying file wasn't independently re-derived byte-for-byte in this pass).

| Table | Caption's own claim | Verdict | Source file | Confidence |
|---|---|---|---|---|
| 3 — Main Results (n=204/20) | "mirrors the offline evaluation common in prior work" | **Offline** | Values closely match `offline_auto_policy_general_n204.json` (`mode: "offline"`) but are not byte-identical — likely an earlier offline run with the same protocol, not this exact file | Caption confirms offline; exact source file not pinned down |
| 4 — Medical Domain (n=42) | (no protocol stated in caption) | **Offline** | `data/benchmark_reports/medical_results.json`, `mode: "offline"` | **Byte-exact**, all 6 systems (e.g. BM25 ROUGE-L 0.35834→0.358, ctx 76.69→76.7; TreeRAG-DFS medical_entity_recall exactly 1.0) |
| 5 — Ablation (n=70) | (no protocol stated in caption) | **Offline** | `data/benchmark_reports/ablation_results.json`, `mode: "offline"` | **Byte-exact** (e.g. cfg_base ROUGE-L 0.49561→0.496, Δ +0.04704→+0.047, ctx 107.36→107.4) |
| 6 — Hyperparameter sensitivity (n=50, seed 42) | "fair generative protocol" | **Online/fair** | `data/benchmark_reports/ablation_sweep_llama.json` | Real per-config latencies of 10–127s (consistent with genuine local-LLM calls) confirm online; not independently re-derived value-by-value |
| 7 — Efficiency Analysis (n=204) | **"offline mode, n = 204"** (stated directly in the table's own caption) | **Offline** | Same family as Table 3 | Caption is self-disclosing; not re-derived |
| 8 — Fair generator-controlled (n=100, seed 42) | "fair generator-controlled" | **Online/fair** | `data/benchmark_reports/online_local_llama_general_v4_n100.json`, `mode: "online"`, `gen_backend: "ollama"` | **Byte-exact** (confirmed independently in B5) |
| 9 — Citation (n=40, fair protocol) | "fair protocol" | Online/fair (caption-stated) | Not independently re-derived in this pass | Caption-only |
| 10 — HotpotQA multi-hop (n=100, fair protocol) | "fair generative protocol" | Online/fair (caption-stated) | Not independently re-derived in this pass | Caption-only |
| 11 — Per-type breakdown (n=100, fair protocol) | "fair protocol, corrected" | Online/fair (caption-stated) | Same underlying run as Table 10 | Caption-only |
| 12 — GovReport (n=40, fair generative protocol) | "fair generative protocol" | Online/fair (caption-stated) | Not independently re-derived in this pass | Caption-only |
| 13 — Robust small-sample stats | References Tables 3/4 (General/Medical, DFS) and the HotpotQA run (Beam) | **General/Medical rows inherit Table 3/4's offline status**; HotpotQA rows are online/fair | n/a (references above) | Inherits from rows above |

## 3. What this means for the paper's core contribution

Tables 3, 4, 5, and 7 — Main Results, Medical Domain, Ablation, and Efficiency — all
use the offline keyword-scoring path, **not** the LLM-based adaptive traversal that is
the paper's central contribution. This was already partially disclosed in the current
draft: Table 3's and Table 7's captions explicitly say so ("mirrors the offline
evaluation," "offline mode, n = 204" — evidently added in an earlier revision round).
**Table 4 and Table 5 do not carry any such disclosure in their captions**, despite
being confirmed byte-exact offline results. Table 4 in particular contains the
medical entity recall = 1.000 result the project notes flag as the paper's most
striking number — it is an offline-keyword-scoring result, not a demonstration of the
LLM-based traversal's medical performance.

Table 6, 8, 9, 10, 11, and 12 — Hyperparameter sensitivity, the headline Fair
Generator-Controlled comparison, Citation, HotpotQA, the per-type breakdown, and
GovReport — are the tables that actually exercise the LLM-based adaptive traversal.
These are the tables that substantiate the paper's core claim.

## 4. Recommended caption additions (for tables lacking disclosure)

- **Table 4** (Medical Domain, n=42): add a footnote matching Table 3/7's existing
  language, e.g. *"Offline keyword-scoring path (Section 4.5); LLM-based node
  relevance judgment is not exercised."*
- **Table 5** (Ablation, n=70): same footnote.
- **Table 13**: clarify in the caption or a footnote that the General/Medical rows
  inherit the offline protocol from Tables 3/4, while only the HotpotQA rows reflect
  the full LLM-based system.

## 5. ★ Table 4 (Medical, n=42) rerun under the fair protocol — the offline result does not hold

Run: `data/benchmark_reports/online_local_llama_medical_b7_20260728_n42.json`, all 6
systems, `--domain medical`, `--mode online`, `--gen-backend ollama --gen-model
llama3.1:8b`, same backend as Table 8. (One interruption mid-run from an external-drive
disconnect; resumed cleanly from the built-in checkpoint, which reuses completed
systems' saved rows rather than re-querying them — BM25/Dense/FlatRAG were unaffected.)

| System | ROUGE-L (fair) | ROUGE-L (Table 4, offline) | Med. Entity Recall (fair) | Med. Entity Recall (Table 4, offline) | Ctx tok (fair) |
|---|---|---|---|---|---|
| BM25 | 0.257 | 0.358 | 0.895 | 1.000 | 76.7 |
| Dense Retrieval | 0.257 | 0.315 | 0.954 | 0.992 | 81.4 |
| FlatRAG | 0.337 | 0.271 | **0.974** | 1.000 | 0‡ |
| RAPTOR | 0.073 | 0.053 | 0.825 | 0.895 | 412.4 |
| PageTree-RAG (DFS) | **0.067** | **0.366** (table's best) | **0.720** (worst) | **1.000** (table's headline) | 4.5 |
| PageTree-RAG (Beam) | 0.117 | 0.265 | 0.893 | 1.000 | 20.1 |

**The result reverses.** Under the offline keyword-scoring path (current Table 4),
PageTree-RAG (DFS) has the best ROUGE-L (0.366) and a perfect medical entity recall
(1.000) — the paper's single most-cited impressive number. Under the fair protocol with
real LLM-based traversal, **PageTree-RAG (DFS) has the *worst* ROUGE-L of all six
systems (0.067) and the *worst* medical entity recall (0.720)**; FlatRAG and Dense
actually lead entity recall (0.974, 0.954). Paired t-test confirms Beam is
significantly worse than BM25/Dense/FlatRAG on ROUGE-L (p<0.001 for all three) under
the fair protocol.

This is not a small effect or noise — it is a complete reversal of the paper's most
prominent domain-specific claim. **The medical_entity_recall=1.000 result cannot be
used to support the paper's core contribution as currently framed**; it is specific to
the offline keyword-scoring fallback path and does not replicate under the LLM-based
adaptive traversal that the paper's title and abstract describe. This should be
reported in the paper directly — either drop the medical entity recall = 1.000 claim
from the abstract/highlights entirely, or reframe Table 4 explicitly as an offline-path
result with the fair-protocol numbers alongside it as the honest comparison.

## 5a. ★ Follow-up: a real bug inflated DFS's collapse — found, fixed, re-measured

The user pushed back on the DFS ROUGE-L=0.067 result ("that can't be right, is there a
bug?") — a fair challenge that turned up a genuine, previously-unknown bug rather than a
pure capability gap.

**Root cause.** 27/42 DFS medical answers were the literal fallback boilerplate ("no
relevant sections found"), traced to `TreeNavigator`'s over-filtering recovery path
(`src/core/error_recovery.py::_recover_critical_nodes`, only wired into DFS —
`beam_search.py` never calls it, which is also why Beam was far less affected). The
recovery heuristic extracted query keywords with a plain regex
(`\b\w{4,}\b`) and required an exact substring match against candidate titles, gated by
`len(title) > 20` (raw character count). Korean is agglutinative — particles attach
directly to nouns — so the regex extracted tokens like "초음파의" (noun + genitive
particle) that never substring-match a title like "초음파 영상의 정의" even though the
core noun ("초음파"/ultrasound) is clearly present in both, and the title-length gate
assumes English-scale titles (a complete, on-topic 4-word Korean title is often under
20 characters). Both defects are English-centric assumptions applied to Korean text.

**Fix, in two passes** (`src/core/error_recovery.py`):
1. First pass: character-length → word-count gate. Real but small effect: 27→25/42.
2. Real fix: added `kiwipiepy` (Korean morphological analyzer, pip-installable, no
   Java/system deps) to extract noun stems (NNG/NNP tags) instead of regex substrings
   for Korean text, keeping the original regex as a fallback for non-Korean text. All
   24 existing `error_recovery` unit/integration tests still pass unmodified.

**Re-measured** (`data/benchmark_reports/online_local_llama_medical_b7_20260730_dfs_kiwi_fixed_n42.json`,
DFS only, same 42 medical questions):

| | Zero-retrieval fallbacks | ROUGE-L | Medical Entity Recall | Ctx tok | Latency |
|---|---|---|---|---|---|
| Before fix | 27/42 (64%) | 0.067 | 0.720 | 4.5 | 41.0s |
| After fix | 13/42 (31%) | 0.087 | 0.835 | 9.8 | 70.1s |

**The bug was real and the fix genuinely helped (27→13 fallbacks, entity recall
0.720→0.835), but it does not overturn §5's reversal.** Even fixed, DFS remains at or
near the bottom of all 6 systems on both ROUGE-L (0.087, still far below FlatRAG's
0.337 or BM25's 0.257) and entity recall (0.835, still below BM25 0.895 / Dense 0.954 /
FlatRAG 0.974). The 13 residual fallbacks include at least one on an *English* document
(numbered headings like "10. CONCLUSION" scored 0.09 confidence against a clearly
on-topic query) — a different, not-yet-diagnosed failure mode, not the Korean-specific
one just fixed. **Honest conclusion: the offline-path Table 4 result
(medical_entity_recall=1.000, DFS best ROUGE-L) still does not replicate under the fair
protocol even after fixing a real, confirmed bug** — the reversal is real, though the
bug inflated its size. Report both the bug-fix and the corrected (still-reversed)
numbers in the paper; do not report only the pre-fix numbers.

## 5b. ★★ Why the reversal happens at all: offline ROUGE-L is inflated by construction, not just by skipping LLM traversal

Pushed further on *why* even the fixed DFS answers score so low (0.087) despite reading,
on inspection, as accurate and on-topic (§5a's non-empty answers). Two things layer
together:

**(1) Answer-length mismatch (same mechanism as B5, replayed here more severely).**
Gold `expected_answer_hint` in the medical benchmark averages **87 characters** — a
terse one-line summary. Every system's actual generated answer is 3–5× longer:
BM25 252.5, Dense 310.8, FlatRAG 238.9, RAPTOR 345.0, PageTree-RAG (DFS, fixed) 368.1,
PageTree-RAG (Beam) 471.4 characters. Since ROUGE-L is F-measure (B5), all systems are
mechanically penalized, and the two PageTree-RAG variants — whose answers are the
*longest* of all six — are penalized hardest. This alone substantially explains DFS's
low score even on the 29 non-empty, factually-correct answers.

**(2) Offline mode's high score is a gold-circularity artifact, not evidence of better
retrieval — direct textual proof.** Checked the *offline* extractive "answer" (raw
node title+summary+page_ref, concatenated, no LLM) against gold for the same questions:

> Gold: *"초음파 영상의 정의와 18세기부터 20세기 중반까지의 기술 발전 과정을 다룸"*
> Offline DFS "answer": *"초음파의 개요 및 역사: **초음파 영상의 정의와 18세기부터
> 20세기 중반까지의 기술 발전 과정을 다룸** [p.3-7] 초음파의 발생 및 장치: ... 초음파의
> 역사: ... [p.6-7]"* — rouge_l = 0.29–0.51 across the three sampled questions.

The gold phrase appears **verbatim, word-for-word**, inside the offline "answer." This
is not a coincidence — it is the direct consequence of how the benchmark's gold was
built (B2): `expected_answer_hint` is generated from, and verified to appear in, the
*same source node's summary text* that offline mode's extractive answer directly
outputs. Offline ROUGE-L is therefore high whenever the right node is anywhere in the
retrieved set, almost independent of whether any real "answering" happened — it is a
second, independent instance of the gold-circularity problem B2 documents for citation
F1, now shown to also inflate offline-mode ROUGE-L specifically. Fair/generative mode
requires the LLM to *paraphrase* that same content into original prose — which is what
a real deployed system does — and paraphrasing mechanically breaks LCS-based overlap
with a gold string lifted verbatim from the source, regardless of factual correctness.
Offline mode's answer is also ~250–350 characters (similar length to the generative
answers, confirmed by direct measurement), so this is **not primarily a length effect
for the offline/online contrast — it's a verbatim-substring effect specific to how the
gold was constructed**, layered on top of the length effect from (1) above.

**Does switching to recall (B5's fix for the general benchmark) rescue PageTree-RAG
here too? Checked directly — no.** Recomputed ROUGE-L Precision/Recall/F on the same
already-generated answers (no new LLM calls; script pattern from B5):

| System | Precision | Recall | F |
|---|---|---|---|
| BM25 | 0.210 | 0.474 | 0.257 |
| Dense | 0.224 | 0.489 | 0.257 |
| FlatRAG | 0.293 | **0.536** | 0.337 |
| RAPTOR | 0.065 | 0.153 | 0.073 |
| PageTree-RAG (DFS) | 0.056 | 0.226 | 0.087 |
| PageTree-RAG (Beam) | 0.077 | 0.334 | 0.117 |

Unlike the general-benchmark result in B5 (where switching to recall flips the ranking
and PageTree-RAG (Beam) wins outright), **on the medical benchmark PageTree-RAG remains
last or near-last under every P/R/F variant.** The metric-choice fix that rescues the
general-benchmark comparison does not apply here — this is not merely a metric
artifact. Combined with (1) and (2) above (length penalty, offline gold-circularity),
the honest picture is: **the medical offline-vs-fair reversal is partly a genuine
metric-construction artifact (offline's inflated score specifically), but the fair-protocol
gap itself is not fully explained away — PageTree-RAG's medical-domain answers
genuinely share less lexical content with the terse, term-specific gold references than
baselines' answers do, under every ROUGE-L variant tested.**

**Combined implication for the paper.** Report all of this, not a single clean story:
(a) Table 4's current offline number is inflated by gold-circularity and should not be
presented as-is without disclosure (§5, §5a, §5b(2)); (b) the fair-protocol reversal is
real and — unlike the general benchmark — does not resolve under a recall-based metric
either, so it should not be waved away as pure measurement artifact; (c) LLM-Judge
(lexical-overlap-independent) tells a friendlier story — PageTree-RAG (Beam)'s
LLM-Judge (0.80) is competitive with every baseline except Dense (0.81) even under the
fair protocol — suggesting the gap may be more about surface lexical match with a
terse, keyword-dense medical gold style than about factual correctness, but this is a
plausible interpretation, not something directly proven here; confirming it would need
a manual read of a sample of DFS/Beam medical answers against gold by a human judge,
which was not done in this pass.

## 6. Rerunning Table 5 (Ablation, n=70) under the fair protocol

Not attempted in this pass — flagged as out of scope for this session given the
already-running ~9-hour B1 rerun. Cost would be comparable to or larger than the
Table 4 rerun (4 configurations × 70 questions with real LLM node scoring), and is a
reasonable next step once the medical rerun above is done.

## Answered vs. unmeasured

- **Answered**: definitive mode identification for Tables 3, 4, 5, 7 (all offline,
  Table 4 and 5 confirmed byte-exact) and 6, 8 (both online/fair, Table 8 byte-exact
  from B5). Caption text confirms 9, 10, 11, 12 as fair-protocol by the authors' own
  prior claim, though not independently re-derived byte-for-byte here.
  A real bug in DFS's over-filtering recovery logic (English-centric keyword/title-length
  heuristics failing on Korean) was found, root-caused, fixed with a Korean morphological
  analyzer, and re-measured (§5a) — the fix reduced but did not eliminate DFS's collapse
  on the medical benchmark, and does not overturn the reversal finding.
- **Not measured**: byte-exact source-file confirmation for Tables 3, 9, 10, 11, 12; a
  fair-protocol ablation rerun (Table 5) — queued as follow-up work, not run in this
  session. The 13 residual post-fix DFS fallbacks include at least one on an English
  document with a distinct, undiagnosed failure mode (numbered section headings) — not
  investigated further here.
