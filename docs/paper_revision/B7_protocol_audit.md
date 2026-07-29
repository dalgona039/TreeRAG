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
- **Not measured**: byte-exact source-file confirmation for Tables 3, 9, 10, 11, 12; a
  fair-protocol ablation rerun (Table 5) — queued as follow-up work, not run in this
  session (lower priority than the medical rerun, which was completed — see §5 above,
  and produced a critical finding: the fair-protocol medical result reverses the
  paper's offline-path headline claim).
