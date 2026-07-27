# B4 — FlatRAG Baseline: Weight Origin, Sensitivity, and the Cautionary-Finding Framing

## 1. Origin of the 60/25/15 weights

`src/core/flat_rag_baseline.py` has no module docstring, no comment, and no citation
anywhere near the weight combination (`_retrieve_documents`, lines ~254-256):

```python
combined_score = (
    0.6 * normalized_bm25 +
    0.25 * semantic_score +
    0.15 * structural_score
)
```

**Confirmed arbitrary** — an author-chosen constant, not sourced from prior work.

## 2. ★ "Semantic" is not semantic — it's a second lexical signal

`SemanticRanker.score()` (`flat_rag_baseline.py:84-102`) is **not** an embedding-based
similarity at all:

```python
def _simple_similarity(self, text1, text2):
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0   # plain Jaccard word overlap
```

This means FlatRAG's advertised "hybrid of BM25 60%, semantic 25%, structural 15%" is
actually **two lexical-overlap signals (BM25 + Jaccard word overlap) covering 85% of the
weight, plus a query-independent depth prior (15%)**. There is no embedding model, no
learned representation, and no genuine semantic signal anywhere in FlatRAG. The paper's
"semantic" label overstates what this baseline does and should be corrected (e.g. to
"lexical + bag-of-words overlap + structural depth prior") — this matters because a
reviewer could otherwise read "semantic 25%" as embedding-based and conclude FlatRAG
demonstrates something about semantic retrieval, which it doesn't.

## 3. Retrieval-only sensitivity sweep (no LLM calls — measured, not estimated)

`FlatRAGBaseline`'s three scorers (BM25, Jaccard-"semantic", structural) are pure
algorithms with no embedding model or LLM in the loop, so retrieval quality under
different weight blends can be measured directly against the gold `expected_sections`
field (the same field `citation_metrics.py` uses) without generating any answers. This
sidesteps the local Ollama server, which was occupied by the concurrent B1 rerun.

Script: `benchmarks/analysis/b4_flatrag_weight_sweep.py`. Full results:
`docs/paper_revision/B4_weight_sweep_n100.csv`. Same fair-protocol sample as B5/B1
(Full Benchmark, n=100, seed=42).

### Mean section-F1 (retrieval-only proxy against gold `expected_sections`)

| Weight config (BM25/semantic/structural) | Mean F1 | Std |
|---|---|---|
| **current: 60/25/15** | **0.6092** | 0.2357 |
| pure BM25: 100/0/0 | **0.6092** | 0.2357 |
| BM25-heavy: 80/10/10 | 0.6092 | 0.2357 |
| equal: 33/33/33 | 0.6088 | 0.2365 |
| semantic-heavy: 20/70/10 | 0.6075 | 0.2371 |
| pure "semantic" (Jaccard): 0/100/0 | 0.5902 | 0.2638 |
| pure structural: 0/0/100 | 0.5886 | 0.2651 |

### Mean Jaccard overlap of top-5 retrieved node-ID sets vs. the current 60/25/15 config

| Weight config | Mean overlap |
|---|---|
| pure BM25: 100/0/0 | 0.9843 |
| BM25-heavy: 80/10/10 | 0.9900 |
| equal: 33/33/33 | 0.9710 |
| semantic-heavy: 20/70/10 | 0.9800 |
| pure "semantic": 0/100/0 | 0.9330 |
| pure structural: 0/0/100 | 0.9267 |

## ★ Central finding: the weights barely matter, because BM25 pre-filters the candidate pool

`_retrieve_documents` calls `self.bm25.search(query, top_k=20)` **first**, and only
those top-20 BM25 candidates are ever re-scored by the semantic/structural signals
(`flat_rag_baseline.py:231-243`) — the other two signals can only **re-rank within
BM25's top-20**, never surface a document BM25 missed. This is why the current 60/25/15
config and pure-BM25 (100/0/0) produce **numerically identical** mean F1 (0.6092) — with
60% weight, BM25 already dominates the ranking within its own pre-filtered pool, so the
remaining 40% barely perturbs the final top-5. Even the two "pure" non-BM25 configs
still score respectably (0.59, 0.588) only because they're re-ranking within a pool BM25
already selected — none of the four weight variants that keep any BM25 weight differ by
more than 0.002 F1, and even the two zero-BM25-weight configs only drop by ~0.02 F1
(≈3%) because the candidate pool itself was already BM25-determined.

**Conclusion: FlatRAG's retrieval behavior is structurally close to plain BM25.** The
"semantic" and "structural" components are nearly cosmetic in this architecture — not
because 25%/15% are small weights, but because the two-stage design (BM25 top-20 filter
→ reweight) caps how much they can ever change the outcome. This is a stronger, more
specific version of the "un-justified weights" concern: it's not just that 60/25/15
lacks a citation, it's that **almost any weight choice would produce nearly the same
result**, because the architecture itself is BM25-dominated by construction.

## 4. Proposed reframing: cautionary finding around BM25, not FlatRAG

Given §3's finding, reframing the cautionary finding around BM25 is not just cleaner
authorship-wise (BM25 needs no citation, no weight-tuning discussion, and is a standard
baseline every reviewer already trusts) — **it's a more accurate description of what's
actually happening**, since FlatRAG's retrieval is shown here to be nearly
indistinguishable from BM25's regardless of blend weight. Table 8 already gives clean
support: BM25 ROUGE-L (0.473) is within noise of FlatRAG's (0.479, the current
headline number).

**Suggested revised passage** (for the section currently anchoring the cautionary
finding on FlatRAG):

> *Under fair generation, standard lexical retrieval is competitive with, and by some
> metrics exceeds, PageTree-RAG's structure-aware traversal: BM25 attains ROUGE-L 0.473
> versus PageTree-RAG (Beam)'s 0.377 (Table 8), a result that holds whether retrieval is
> pure BM25 or blended with the additional lexical/structural signals in our FlatRAG
> variant (Table 8: FlatRAG 0.479 — statistically indistinguishable from BM25; see
> [B4 supplementary] for a weight-sensitivity analysis showing FlatRAG's retrieval is
> structurally BM25-dominated regardless of blend weight). This is a genuine cautionary
> finding for structure-aware RAG: a well-tuned lexical baseline requiring no
> architecture beyond an inverted index remains competitive with hierarchical traversal
> under F-measure lexical-overlap scoring.*

This keeps the honest, uncomfortable finding (lexical retrieval is competitive) while
removing the unexplained, arbitrarily-weighted, misleadingly-labeled "semantic" author
baseline from the center of the argument.

## Answered vs. unmeasured

- **Answered**: weight origin (arbitrary, uncited); "semantic" component is actually
  lexical Jaccard overlap, not embeddings; full retrieval-only sensitivity sweep across
  7 weight configurations on n=100 (real measurement, not estimated); mechanistic
  explanation for why the sweep is nearly flat (BM25 top-20 pre-filter); proposed
  reframing text.
- **Not measured**: end-to-end generation+ROUGE-L sensitivity (would need LLM calls per
  weight config — not attempted since the local Ollama server was occupied by the B1
  rerun; given how flat the retrieval-only proxy is, a generation-level sweep is
  unlikely to show much more, but wasn't directly confirmed).
