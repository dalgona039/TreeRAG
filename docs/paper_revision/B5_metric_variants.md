# B5 — Metric Variant Audit (ROUGE-L / BERTScore: recall vs. precision vs. F)

## 1. What the code actually computes

Source: `benchmarks/metrics/text_similarity.py`.

| Metric | Paper's prose claim | What the code computes | Library | Config |
|---|---|---|---|---|
| ROUGE-L | "longest-common-subsequence **recall**" | **F-measure** — `result["rougeL"].fmeasure` (`text_similarity.py:64`) | `rouge_score` (Google's `rouge-score`, v0.1.2) | Custom `_MultilingualTokenizer` (`text_similarity.py:47-53`) because the library's default tokenizer strips non-ASCII and zeroes out Korean/CJK text. No stemming (Porter stemmer not used in this custom tokenizer path). |
| BERTScore | "contextual embedding **precision**" | **F1** — `f1.mean().item()` when the real library loads (`text_similarity.py:121`); **falls back to a hand-written multiset token-F1 proxy** (`_token_f1`, also F, not P) when `bert_score` can't be imported (`text_similarity.py:104-123`) | `bert-score` when available, else in-house proxy | Model: `klue/roberta-base` for `lang="ko"`, `roberta-base` for other langs (`text_similarity.py:117`). No baseline rescaling call found anywhere. |

**Verdict: both metrics are described in the paper as one-sided (recall / precision), but the
code computes the two-sided F-measure for both.** This is not a subtle wording slip — recall
and precision are the two quantities the paper explicitly contrasts to explain *why* ROUGE-L
and BERTScore should behave differently, and the code uses neither at its stated setting.

### 1a. Is "BERTScore" in the stored results even real BERTScore?

`bert-score`, `sentence-transformers`, and `torch` are **not** in `requirements.txt` and are
**not installed** in this project's `.venv` (checked directly). The `bertscore_f1()` function
therefore silently takes the `except Exception: return _token_f1(...)` branch
(`text_similarity.py:122-123`) — a lexical multiset-overlap F1 with no embeddings, no
"contextual" anything. Corroborating evidence: BERTScore values in the stored fair-protocol
result file range 0.22–0.53 (see below), which is far below the ~0.85–0.95 floor real
RoBERTa-based BERTScore typically produces even for weak matches (raw, unrescaled BERTScore
rarely drops below ~0.8 because cosine similarity in embedding space is compressed). This
confirms with high confidence that **every "BERTScore" number in the paper is actually the
token-F1 lexical proxy**, not contextual embeddings at all.

## 2. Recomputation on stored fair-protocol answers (n=100)

No LLM calls were made. I loaded the already-stored generative fair-protocol result file
`data/benchmark_reports/online_local_llama_general_v4_n100.json` (`gen_backend: ollama`,
`gen_model: llama3.1:8b`, 6 systems × 100 questions) and joined each answer to its gold
`expected_answer_hint` from `benchmarks/datasets/full_benchmark.json` by `question_id`, then
recomputed ROUGE-L and the BERTScore-proxy with precision, recall, and F reported separately,
using the exact tokenizer/scorer the pipeline uses
(script: `benchmarks/analysis/b5_recompute_metric_variants.py`, full per-question CSV:
`docs/paper_revision/B5_metric_variants_n100.csv`).

**Sanity check — this file is confirmed to be the source of the paper's Table 8.** The
recomputed F-measures match Table 8 in `TreeRAG_TIST_ACM.docx` to 3 decimal places for all 6
systems on both ROUGE-L and BERTScore:

| System | Table 8 ROUGE-L | Recomputed F | Table 8 BERTScore | Recomputed F |
|---|---|---|---|---|
| BM25 | 0.473 | 0.4726 | 0.530 | 0.5303 |
| Dense Retrieval | 0.435 | 0.4347 | 0.486 | 0.4861 |
| FlatRAG | 0.479 | 0.4791 | 0.528 | 0.5275 |
| RAPTOR | 0.384 | 0.3842 | 0.437 | 0.4367 |
| PageTree-RAG (DFS) | 0.340 | 0.3396 | 0.389 | 0.3893 |
| PageTree-RAG (Beam) | 0.377 | 0.3771 | 0.439 | 0.4389 |

### Answer length (words), mean ± population std, n=100

| System | Mean len | Std |
|---|---|---|
| BM25 | 31.1 | 25.5 |
| Dense | 31.6 | 28.8 |
| FlatRAG | 34.1 | 26.6 |
| RAPTOR | 34.8 | 32.8 |
| PageTree-RAG (DFS) | 52.3 | 35.3 |
| PageTree-RAG (Beam) | 54.6 | 37.4 |

PageTree-RAG's answers are ~55–75% longer on average than every baseline's. Under an
F-measure metric, extra length that isn't matched in the reference is pure precision loss
with no recall benefit once the reference is already covered — this mechanically penalizes
longer answers, independent of answer quality.

### ROUGE-L: Precision / Recall / F, n=100

| System | Precision | Recall | F |
|---|---|---|---|
| BM25 | 0.5368 | 0.5242 | 0.4726 |
| Dense | 0.4961 | 0.4662 | 0.4347 |
| FlatRAG | 0.5120 | 0.5511 | 0.4791 |
| RAPTOR | 0.4368 | 0.4462 | 0.3842 |
| PageTree-RAG (DFS) | 0.2903 | 0.5287 | 0.3396 |
| **PageTree-RAG (Beam)** | 0.3149 | **0.5821** | 0.3771 |

### BERTScore-proxy (token-F1 lexical overlap — see §1a): Precision / Recall / F, n=100

| System | Precision | Recall | F |
|---|---|---|---|
| BM25 | 0.5889 | 0.6004 | 0.5303 |
| Dense | 0.5472 | 0.5293 | 0.4861 |
| FlatRAG | 0.5557 | 0.6180 | 0.5275 |
| RAPTOR | 0.4879 | 0.5138 | 0.4367 |
| PageTree-RAG (DFS) | 0.3301 | 0.6135 | 0.3893 |
| **PageTree-RAG (Beam)** | 0.3636 | **0.6839** | 0.4389 |

## ★ Central finding — the ranking is a direct artifact of which variant is chosen

Under **F** (what the code and Table 8 actually report): FlatRAG leads ROUGE-L (0.479),
BM25 leads BERTScore-proxy (0.530). This is the paper's current "cautionary finding" —
plain lexical/flat retrieval beats PageTree-RAG on lexical metrics.

Under **recall** (what the paper's prose *claims* ROUGE-L measures): **PageTree-RAG (Beam)
has the highest ROUGE-L recall of all six systems (0.582), and the highest BERTScore-proxy
recall (0.684)** — beating FlatRAG (0.551 / 0.618) and BM25 (0.524 / 0.600) outright.

So the paper's own stated metric definition, applied literally, **reverses the headline
result**. The F-measure ranking that Table 8 actually reports is real and shouldn't be
hidden, but it is a consequence of PageTree-RAG's longer answers costing precision, not of
PageTree-RAG retrieving worse content — recall says the opposite. This is exactly the kind
of result the task asked to surface even though it complicates the paper's argument: **the
ROUGE-L reversal is not an inherent property of the retrieval systems; it is a byproduct of
(a) which P/R/F variant is used, compounded by (b) PageTree-RAG's answers being ~1.5–1.75×
longer than baselines' with no length normalization anywhere in the scoring pipeline.**

This has two viable framings for the paper, both defensible, neither is "hide it":
1. Keep F as the headline metric (already what's reported), but fix the prose so it says
   F-measure, not recall/precision, and add the recall table above as a companion result
   showing PageTree-RAG's retrieval is recall-optimal but length-inflated — turning the
   "cautionary finding" into a nuanced precision/length story instead of a straightforward
   loss.
2. Report all three (P/R/F) for both metrics as a robustness table, exactly like the one
   above, and let the reader see the full picture. Given how much of the paper's argument
   currently rests on the single F-measure number, this is the safer choice for peer review.

## Answered vs. unmeasured

- **Answered**: ROUGE-L/BERTScore are F, not R/P as claimed (code-certain). BERTScore is a
  lexical token-F1 proxy, not contextual embeddings, in the stored results (high-confidence
  from missing deps + score-range evidence, not a literal per-run log). Recomputed P/R/F for
  all 6 systems, n=100, on the exact file that produced Table 8.
- **Not measured**: Whether the same reversal holds on the HotpotQA/GovReport tables (Table 2,
  the "System / ROUGE-L / BERTScore / LLM-Judge / HotpotQA R-L / Latency" table) — those use a
  different stored result file and were not recomputed here; recommend running the same script
  against the HotpotQA-specific result file if one exists with per-question answer+gold pairs.
  Real (non-proxy) BERTScore was not computed anywhere in this audit since `bert-score`/`torch`
  are not installed in this environment — installing them and re-scoring would be the strongest
  possible fix but is a nontrivial dependency addition, not attempted here.
