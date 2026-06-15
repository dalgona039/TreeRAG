# KCI Execution Prompt

> **Usage**: Paste the prompt below into a new Claude conversation with this TreeRAG project folder attached.  
> The prompt is self-contained — Claude will read the codebase and execute each phase without further instruction.

---

```
You are a research engineering assistant helping upgrade the TreeRAG project
(located in the connected folder) to meet KCI (Korean Citation Index) academic
publication standards.

## Project Context

TreeRAG is a hierarchical document RAG system built on FastAPI + Next.js. It
converts PDFs into JSON trees via LLM (Gemini), then traverses the tree using
DFS or Beam Search to answer user queries. The system is fully implemented
(PHASE 1-3 complete, 469 tests passing), but the experimental validation is
insufficient for academic publication.

Key files to read first:
- README.md — full system overview
- RESEARCH_OVERVIEW.md — academic framing and paper structure
- KCI_PUBLICATION_PLAN.md — the detailed gap analysis and plan (read this first)
- src/core/reasoner.py — main inference engine
- src/core/tree_traversal.py — DFS traversal
- src/core/beam_search.py — Beam Search traversal
- src/core/flat_rag_baseline.py — existing FlatRAG baseline
- benchmarks/run_real_evaluation.py — existing evaluation runner (incomplete)
- benchmarks/metrics/ — existing metric modules
- scripts/ablation_study.py — existing ablation runner (not yet executed)
- data/indices/ — pre-built document indices (use these for experiments)

## Your Mission

Implement the four phases below IN ORDER. Before starting each phase, read the
relevant existing files to avoid duplicating code. After completing each phase,
run the existing test suite to confirm nothing is broken:
  conda activate medireg && pytest -q --tb=short

---

## PHASE A — Dataset Construction

### A-1: Auto Q&A Generator
Create `benchmarks/datasets/auto_qa_generator.py`.

This module must:
1. Accept a PageIndex JSON file (from data/indices/) as input.
2. Call Gemini API (via `src/config.Config`) with the following prompt template
   to generate questions for each document:

   ```
   You are a QA dataset creator for a RAG evaluation benchmark.
   Given the following document tree structure, generate {n} diverse questions
   that test different retrieval scenarios.

   Document tree:
   {tree_json}

   Generate questions in three categories:
   - factual (5): Single-section lookup, e.g. "What is X?"
   - multi_hop (3): Requires combining 2+ sections, e.g. "How does A relate to B?"
   - comparative (2): Requires comparison, e.g. "What are the differences between X and Y?"

   For each question provide:
   - question: the question string
   - expected_sections: list of node IDs that contain the answer
   - expected_answer_hint: a brief expected answer (1-2 sentences)
   - difficulty: "easy" | "medium" | "hard"
   - category: "factual" | "multi_hop" | "comparative"

   Respond in JSON only:
   {"questions": [...]}
   ```

3. Run this against ALL existing index files in data/indices/.
4. Merge all results into `benchmarks/datasets/full_benchmark.json` with schema:
   ```json
   {
     "version": "2.0",
     "total_questions": <N>,
     "documents": ["doc1", "doc2", ...],
     "questions": [
       {
         "question_id": "auto_001",
         "document_id": "<doc_name>",
         "question": "...",
         "expected_sections": ["node_id_1"],
         "expected_answer_hint": "...",
         "difficulty": "medium",
         "category": "factual"
       }
     ]
   }
   ```
5. Target: minimum 50 questions total across all documents.
6. Print a summary table after generation (doc name, question count, categories).

### A-2: Validate Dataset
Write a simple validation function that checks:
- No duplicate questions
- All document_ids resolve to an existing index file
- Each question has all required fields
Print a PASS/FAIL report.

---

## PHASE B — Baseline Expansion

### B-1: BM25 Retriever
Create `src/core/bm25_baseline.py`.

Requirements:
- Use the `rank_bm25` library (install if needed: pip install rank-bm25).
- Class `BM25Retriever` with the same interface as `FlatRAGBaseline`:
  - `__init__(self, index: dict)` — accepts a parsed PageIndex JSON dict
  - `retrieve(self, query: str, top_k: int = 10) -> list[dict]` — returns ranked
    node dicts, each with keys: `title`, `summary`, `page_ref`, `score`
- Tokenization: split on whitespace + punctuation (support Korean).
- Index all nodes (title + summary concatenated) at init time.
- Do NOT call any external API — pure keyword matching only.
- Add 10 unit tests in `tests/test_bm25_baseline.py`.

### B-2: Dense Retrieval Baseline
Create `src/core/dense_retrieval_baseline.py`.

Requirements:
- Use `sentence-transformers` library with model
  `jhgan/ko-sroberta-multitask` for Korean-primary documents.
  Fallback to `intfloat/multilingual-e5-base` if unavailable.
- Install if needed: pip install sentence-transformers faiss-cpu
- Class `DenseRetriever` with same interface as BM25Retriever above.
- Build FAISS flat index (IndexFlatIP) at init time from node embeddings.
- `retrieve()` returns top-k nodes by cosine similarity.
- Cache embeddings to disk at `data/indices/{doc_hash}_dense_index.pkl`
  so rebuilding is skipped on repeated runs.
- Add 10 unit tests in `tests/test_dense_retrieval_baseline.py`.
  Use mock embeddings (random vectors) so tests run without GPU.

---

## PHASE C — Automated Evaluation Pipeline

### C-1: Text Similarity Metrics
Create `benchmarks/metrics/text_similarity.py`.

Implement these three functions (all return float 0.0–1.0):

```python
def rouge_l_score(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 using the rouge-score library."""
    # pip install rouge-score

def bertscore_f1(hypothesis: str, reference: str, lang: str = "ko") -> float:
    """Compute BERTScore F1 using bert-score library."""
    # pip install bert-score
    # Use model: klue/roberta-base for Korean, roberta-base for English

def exact_match(hypothesis: str, reference: str) -> float:
    """Return 1.0 if normalized strings match exactly, else 0.0."""
    # Normalize: lowercase, strip punctuation/whitespace
```

Also implement `batch_evaluate(hypotheses, references, metrics)` that runs
all requested metrics in parallel and returns a dict of lists.

Add 15 unit tests in `tests/test_text_similarity.py` covering edge cases
(empty strings, perfect match, partial match, Korean text).

### C-2: LLM-as-Judge Evaluator
Create `benchmarks/metrics/llm_judge.py`.

```python
class GeminiJudge:
    """
    Uses Gemini to score RAG answers on three axes (0-5 each):
    - faithfulness: Is the answer grounded in the source documents?
    - relevance: Does the answer directly address the question?
    - completeness: Does it cover all key aspects of the answer?
    """

    JUDGE_PROMPT = """
You are an expert evaluator for a document QA system. Score the following answer
on three criteria, each from 0 to 5:

Question: {question}
Source Context: {context}
System Answer: {answer}
Expected Answer Hint: {expected}

Scoring criteria:
- faithfulness (0-5): 5 = fully grounded in source, 0 = fabricated
- relevance (0-5): 5 = directly answers the question, 0 = irrelevant
- completeness (0-5): 5 = covers all key points, 0 = missing core information

Respond in JSON only:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>,
  "reasoning": "<one sentence>"}}
"""

    def score(self, question, context, answer, expected) -> dict:
        # Call Config.CLIENT, parse JSON response
        # Return normalized scores (divide by 5) plus reasoning
        # Handle JSON parse errors gracefully (return None scores)
```

Add 12 unit tests with mocked Gemini responses.

### C-3: Complete the Evaluation Runner
Rewrite `benchmarks/run_real_evaluation.py` to:

1. Accept CLI args: `--dataset`, `--systems`, `--output`, `--use-llm-judge`
2. Load `full_benchmark.json` from PHASE A.
3. For each system in `--systems` (choices: bm25, dense, flatrag, treerag_dfs,
   treerag_beam), run each question through the system and collect:
   - answer text
   - retrieved node count
   - context token count (estimate: chars / 4)
   - latency in seconds
4. Score each answer with ROUGE-L, BERTScore, and optionally LLM-judge.
5. Save full results to `data/benchmark_reports/evaluation_{timestamp}.json`.
6. Print a formatted comparison table to stdout:

   ```
   ┌─────────────────┬─────────┬───────────┬───────────┬──────────┬────────┐
   │ System          │ ROUGE-L │ BERTScore │ LLM-Judge │ Latency  │ CTX(K) │
   ├─────────────────┼─────────┼───────────┼───────────┼──────────┼────────┤
   │ BM25            │  0.XXX  │   0.XXX   │   0.XX    │  X.XXs   │  XX.X  │
   │ Dense Retrieval │  0.XXX  │   0.XXX   │   0.XX    │  X.XXs   │  XX.X  │
   │ FlatRAG         │  0.XXX  │   0.XXX   │   0.XX    │  X.XXs   │  XX.X  │
   │ TreeRAG-DFS     │  0.XXX  │   0.XXX   │   0.XX    │  X.XXs   │  XX.X  │
   │ TreeRAG-Beam    │  0.XXX  │   0.XXX   │   0.XX    │  X.XXs   │  XX.X  │
   └─────────────────┴─────────┴───────────┴───────────┴──────────┴────────┘
   ```

7. Run statistical significance tests (use existing
   `benchmarks/metrics/statistical_tests.py`) between TreeRAG-Beam and each
   baseline. Print p-values.

---

## PHASE D — Ablation Study & Visualization

### D-1: Run Ablation Study
Update `scripts/ablation_study.py` to test these 4 configurations against the
full benchmark dataset:

| Config ID | Beam Search | Contextual Compression | Reference Resolver |
|-----------|-------------|------------------------|-------------------|
| cfg_base  | ✗ (DFS)     | ✗                      | ✗                 |
| cfg_beam  | ✓           | ✗                      | ✗                 |
| cfg_beam_compress | ✓   | ✓                      | ✗                 |
| cfg_full  | ✓           | ✓                      | ✓                 |

For each config, use `TreeRAGReasoner` with appropriate flags and collect
ROUGE-L, BERTScore, latency, and context size. Save to
`data/benchmark_reports/ablation_results.json`.

### D-2: Generate LaTeX Tables
Update `scripts/generate_paper_tables.py` to read the results from PHASE C
and D and output ready-to-paste LaTeX code for:

**Table 1** (Main comparison): ROUGE-L, BERTScore, LLM-Judge, Latency, CTX
**Table 2** (Ablation): Config, ROUGE-L, Delta vs Full, CTX Reduction
**Table 3** (Efficiency by doc size): <50p, 50-100p, >100p — CTX reduction

Include \\textbf{} on the best number per column. Include p-values as footnotes.
Save output to `data/benchmark_reports/paper_tables.tex`.

### D-3: Generate Figures
Update `scripts/plot_results.py` to generate:

1. `figure_1_comparison.png`: Grouped bar chart, systems × metrics
   (ROUGE-L and BERTScore side by side per system)
2. `figure_2_ablation.png`: Horizontal bar chart of ablation configs, 
   sorted by ROUGE-L, with delta labels
3. `figure_3_efficiency.png`: Scatter plot of Latency vs ROUGE-L for all
   systems (bubble size = context size)

Use matplotlib with seaborn style. Save to `data/benchmark_reports/figures/`.
DPI=300, format=PDF (for LaTeX inclusion).

---

## Completion Criteria

After all phases are complete, verify:

1. `pytest -q --tb=short` passes with 490+ tests (was 469 before).
2. `python benchmarks/run_real_evaluation.py --dataset benchmarks/datasets/full_benchmark.json --systems all` runs end-to-end and produces a results table.
3. `python scripts/ablation_study.py` runs and saves ablation_results.json.
4. `python scripts/generate_paper_tables.py` produces paper_tables.tex with real numbers.
5. All three figures are generated in data/benchmark_reports/figures/.

Report the final numbers from the comparison table, the p-values, and the
ablation deltas so the user can fill them into the paper.
```
