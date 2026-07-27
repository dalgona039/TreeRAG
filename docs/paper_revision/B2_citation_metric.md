# B2 — Citation Metric Definitions and Gold-Circularity Audit

## 1. Exact definitions (from code)

Source: `benchmarks/metrics/citation_metrics.py`, called from `benchmarks/run_exp1_citation.py`
(the actual driver for Table 9 — **not** `run_real_evaluation.py`).

```python
def citation_availability(nodes) -> bool:
    return any(bool(_page_ref(n)) for n in nodes)          # checks node["page_ref"]

def section_citation_f1(nodes, expected_sections) -> float:
    r_ids = [_node_id(n) for n in nodes]                    # checks node["id"]
    e_ids = [str(e) for e in expected_sections]
    precision = |{r : r matches some e}| / |r_ids|
    recall    = |{e : e matches some r}| / |e_ids|
    f1 = 2PR/(P+R)
    # match(r, e) := r == e or r.startswith(e+".") or e.startswith(r+".")   (ancestor/descendant dot-path)
```

Both functions are called on the exact same `nodes` object returned directly from
retrieval (`run_exp1_citation.py:113-116`: `answer, nodes = evaluator.run_system(...)`).
**Neither function ever inspects `answer`.** Confirmed by reading the call site — `avail`
and `cit_f1` are computed from `nodes` alone, `answer` is only used afterward for ROUGE-L.

- **Availability**: does any retrieved node object have a non-empty `page_ref` field?
- **F1**: do the retrieved nodes' `id` fields match the gold `expected_sections` list
  (ancestor/descendant dot-path), precision/recall/F1 over the retrieved-vs-gold ID sets.

Both are retrieval-stage, structural checks over the same node objects — they differ only
in which field of those objects they read (`page_ref` vs `id`), not in "generated text vs.
retrieved evidence" as currently described.

## 2. ★ The paper's caption is factually wrong about Availability

Current caption (Table 9, `TreeRAG_TIST_ACM.docx`): *"Citation Availability asks whether
**the generated answer text** contains a parseable page reference, whereas Citation F1
compares the section identifiers attributable to the retrieved evidence against the gold
supporting sections."*

This is **incorrect for Availability** (confirmed false by the code above — it never
touches the generated answer, only the retrieved node objects) and **correct for F1**.
The caption's real/false distinction should be corrected to: both metrics inspect the
same retrieved node objects; Availability checks the `page_ref` field, F1 checks the `id`
field against gold section IDs.

## 3. The FlatRAG paradox (Availability=0.000, F1=0.570), fully explained

`_run_flatrag` in `benchmarks/run_real_evaluation.py:230-243` builds citation-metric nodes as:

```python
nodes = [{"id": d} for d in meta.get("retrieved_docs", [])]
```

— only the `"id"` key is ever set; `"page_ref"` is never populated for FlatRAG's citation
nodes anywhere in this code path. So `citation_availability` is unconditionally `False` for
every FlatRAG question → **Availability = 0.000 by construction**, not because FlatRAG
retrieves bad evidence.

`meta["retrieved_docs"]` itself comes from `FlatRAGBaseline.query()`
(`src/core/flat_rag_baseline.py:210`): `'retrieved_docs': [node['id'] for node in retrieved_nodes]`
— these are **real internal tree node IDs** (FlatRAG's hybrid BM25/semantic/structural
scorer operates over the same PageIndex tree as every other system, see B4), not raw
document filenames. Verified against an actual index tree
(`data/indices/61dd7aa0_s41598-026-41649-2_reference_index.json`): node IDs are genuine
hierarchical dot-paths (`CH1`, `SEC1.1`, `ART1.1.1`, ...), matching the format
`section_citation_f1`'s ancestor/descendant matcher expects. So FlatRAG's retrieved node
IDs really can, and often do, match `expected_sections` at the section level — producing a
real, non-trivial F1 — while the same node objects simply never got a `page_ref` value
attached in this particular code path. **Conclusion: not a bug in the metric, but an
inconsistency in how FlatRAG's citation-purpose node dicts are constructed** (`id` populated,
`page_ref` never populated), which the caption should explain instead of the currently
incorrect "generated text" framing.

## 4. Gold-circularity: confirmed real for the Full Benchmark

Checked `benchmarks/datasets/full_benchmark.json`'s 204 questions:
`expected_sections` values are drawn from `{SEC0, SEC1, ROOT, CH*, SEC*.*, ART*.*.*,
document, doc, ...}` — i.e., **literal node IDs from the source PageIndex tree the
question-generation LLM was shown**, confirmed directly against the actual index files
(e.g. `data/indices/hotpotqa_hp_sample_001_index.json` really does have nodes with
`id: "ROOT"`, `id: "SEC0"`, `id: "SEC1"`). This confirms the concern exactly as
stated: gold supporting section = the exact tree node the question was generated from,
which is PageTree-RAG's native retrieval unit. Chunk-based systems (BM25, Dense) don't
operate over this ID space at all in the same way structurally, but FlatRAG and RAPTOR
do retrieve against these same tree nodes (per §3), so the circularity specifically
privileges tree-node-granularity retrieval (PageTree-RAG, FlatRAG, RAPTOR) over
raw-chunk retrieval (BM25, Dense) — not uniquely PageTree-RAG as the framing might
suggest, though PageTree-RAG is still the closest possible match (its traversal literally
targets the same node the question was written from).

## 5. HotpotQA re-measurement (independent gold) — not done in this session

HotpotQA's `supporting_facts` are independent of this repo's tree structure, so citation
F1 recomputed there would not share the circularity in §4. **This requires a live rerun**
(6 systems × n=100 HotpotQA questions with citation metrics) and was not attempted in
this session — the local Ollama server is currently occupied by the ~9-hour B1
full-benchmark rerun, and `run_exp1_citation.py` would need to be adapted to score
against HotpotQA's own supporting-facts field rather than `expected_sections`, which
itself takes some implementation work (checking `run_exp1_citation.py`'s dataset loader
would be a prerequisite step). Flagged as the single highest-value follow-up for
defending or reframing the citation F1 = 0.757 headline number.

## Answered vs. unmeasured

- **Answered**: exact formal definitions of both metrics (code-certain); confirmed neither
  touches generated answer text; fully explained the FlatRAG Availability/F1 paradox;
  confirmed the caption's Availability description is factually wrong and should be
  corrected; confirmed gold-circularity is real and precisely characterized (applies to
  tree-node-granularity retrievers: PageTree-RAG, FlatRAG, RAPTOR — not BM25/Dense).
- **Not measured**: HotpotQA-based independent citation F1 re-measurement (needs a live
  rerun plus adapting the driver to HotpotQA's `supporting_facts` field instead of
  `expected_sections`); whether the ranking (PageTree-RAG Beam highest) holds under that
  independent gold.
