# ACM-Level Upgrade Execution Prompt

> **Usage**: Open a new conversation with this TreeRAG folder attached, then
> paste the prompt below. The prompt is fully self-contained.
> Run phases in order — each phase builds on the previous one.

---

```
You are a research engineering assistant helping upgrade the TreeRAG project
to ACM publication quality (targeting ACM CIKM 2026 or ACM TIST).

## Read These First (mandatory before writing any code)

1. README.md — full system overview
2. RESEARCH_OVERVIEW.md — academic framing
3. KCI_PUBLICATION_PLAN.md — previous gap analysis (KCI-level work already done)
4. KCI_RUN_RESULTS.md — what was implemented and current test status
5. src/core/reasoner.py — main inference engine
6. src/core/flat_rag_baseline.py — existing FlatRAG baseline interface
7. src/core/beam_search.py — Beam Search traversal
8. benchmarks/run_real_evaluation.py — evaluation runner from previous phase
9. benchmarks/datasets/full_benchmark.json — existing Q&A dataset (70 questions)
10. data/indices/ — list all files here; these are the pre-built document trees

After reading, confirm your understanding by printing a one-paragraph summary
of what TreeRAG does before starting any implementation.

## Environment Constraints (critical)

- The sandbox MAY block Gemini API, HuggingFace model downloads, and
  some external HTTP calls. PyPI package installs via pip generally work.
- Every module you write MUST follow this pattern:
    try:
        result = call_real_backend(...)
    except Exception:
        result = deterministic_offline_fallback(...)
  The offline fallback must be deterministic and meaningful (not random),
  so results are reproducible even without network access.
- Python version in sandbox may be 3.10. Do NOT use Python 3.12+ syntax
  (no f-string `=` specifiers like f"{x=}", no match/case statements,
  no PEP 695 type aliases). Use explicit `.format()` or concatenation.
- After completing each phase, run:
    pytest -q --tb=short 2>&1 | tail -5
  and confirm the test count did not decrease from its starting value.
- Do NOT modify any existing source file unless you have read it first
  and the change is strictly necessary to fix a compatibility issue.

## Mission Overview

The primary scientific claim of TreeRAG is:
  "Preserving the original hierarchical document structure (top-down) yields
   better retrieval precision and auditability than bottom-up abstractive
   summarization (RAPTOR) or flat chunk retrieval (Dense/BM25), especially
   in multi-hop reasoning and domain-specific (medical/legal) documents."

Your job is to build the experimental evidence that proves this claim at
ACM peer-review level. There are five phases below. Implement them in order.

---

## PHASE 1 — RAPTOR Baseline Integration

RAPTOR (ICLR 2024, Sarthi et al.) is the closest competing method and the
most important baseline for the ACM submission. Reviewers will ask about it.

### 1-1: Install and wrap RAPTOR

First try: pip install raptor
If unavailable: pip install git+https://github.com/parthsarthi03/raptor.git
If both fail: implement RaptorFallback below.

Create `src/core/raptor_baseline.py` with this interface:

```python
class RaptorBaseline:
    """
    Wraps RAPTOR for comparison with TreeRAG.
    Builds a bottom-up clustering tree from document text,
    then retrieves using RAPTOR's tree traversal.

    Falls back to RaptorFallback if the library is unavailable.
    """

    def __init__(self, document_text: str, document_name: str):
        """
        Args:
            document_text: full plain text of the document (concatenated pages)
            document_name: identifier string
        """

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Returns list of dicts with keys: title, summary, page_ref, score.
        Must match the interface of FlatRAGBaseline.retrieve().
        """

    def answer(self, query: str) -> dict:
        """
        Returns: {
            "answer": str,
            "source_nodes": list[dict],
            "context_tokens": int,
            "latency_ms": float
        }
        """


class RaptorFallback:
    """
    Offline RAPTOR approximation when the library is unavailable.

    Algorithm:
    1. Split document_text into chunks of ~300 chars.
    2. Group chunks into clusters of 3 using a simple round-robin (deterministic).
    3. For each cluster, create a synthetic summary node by concatenating the
       first 80 chars of each chunk member.
    4. Build a 2-level tree: root → cluster_nodes → chunk_nodes.
    5. For retrieval, score each node by character-level overlap with query,
       return top_k by score.

    This approximates RAPTOR's bottom-up structure without requiring the library.
    It is intentionally simple so that real RAPTOR (if available) always wins.
    """
```

### 1-2: Extract plain text from existing PDFs

Create `src/utils/pdf_text_extractor.py`:

```python
def extract_text(pdf_path: str) -> str:
    """
    Extract full plain text from a PDF using pypdf (already in requirements).
    Returns concatenated text of all pages, with page markers:
    "--- PAGE 1 ---\n{text}\n--- PAGE 2 ---\n..."
    """
```

Run this on all PDFs in `data/raw/` and save outputs to
`data/raw_text/{filename}.txt` so RAPTOR has plain text input.

### 1-3: Add RAPTOR to the evaluation runner

In `benchmarks/run_real_evaluation.py`, add "raptor" as a valid system name.
When running raptor, use the extracted plain text from 1-2 as input.

### 1-4: Add RAPTOR comparison table section to generate_paper_tables.py

The main comparison table (Table 1 in the paper) must now include:

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) | CTX(K) |
|--------|---------|-----------|-----------|------------|--------|
| BM25 | | | | | |
| Dense Retrieval | | | | | |
| FlatRAG | | | | | |
| RAPTOR | | | | | ← NEW
| TreeRAG-DFS | | | | | |
| **TreeRAG-Beam** | | | | | |

### 1-5: Write the differentiation analysis

Create `benchmarks/analysis/raptor_vs_treerag.py` that, given evaluation
results for both systems, automatically computes and prints:

- Mean ROUGE-L difference: TreeRAG - RAPTOR
- Mean latency difference
- Win rate: % of questions where TreeRAG beats RAPTOR on ROUGE-L
- Category breakdown: factual vs multi_hop vs comparative win rates
- Page citation availability: % of answers with [doc, p.X] (TreeRAG only)

This data goes directly into the Discussion section of the paper.

Add 10 unit tests in `tests/test_raptor_baseline.py` using mock inputs.

---

## PHASE 2 — Multi-Hop Benchmark (HotpotQA)

TreeRAG's tree traversal should excel on multi-hop questions. This phase
provides the standard benchmark to prove it.

### 2-1: HotpotQA Loader

Create `benchmarks/datasets/hotpotqa_loader.py`:

```python
def load_hotpotqa_subset(n: int = 100, seed: int = 42) -> list[dict]:
    """
    Loads n questions from HotpotQA dev set.

    Primary path: download JSON from
      http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
    (This is a direct HTTP download, no auth needed.)

    Fallback path (if download blocked): load from
      benchmarks/datasets/hotpotqa_sample.json
    (We will create this sample file in step 2-2.)

    Filters to questions where:
      - type == "comparison" OR type == "bridge"  (multi-hop only)
      - answer length < 100 chars

    Returns list of dicts:
    {
      "question_id": str,
      "question": str,
      "answer": str,
      "type": "comparison" | "bridge",
      "supporting_facts": [{"title": str, "sent_id": int}],
      "context": [{"title": str, "sentences": [str]}]
    }
    """

def convert_to_benchmark_format(hotpotqa_items: list[dict]) -> dict:
    """
    Converts HotpotQA items to the same schema as full_benchmark.json
    so run_real_evaluation.py can process them without modification.

    Note: HotpotQA context is provided as sentence lists.
    Concatenate them into a pseudo-document and build a simple flat
    PageIndex JSON so TreeRAG's indexer can process it.
    """
```

### 2-2: Create offline fallback sample

Create `benchmarks/datasets/hotpotqa_sample.json` with 20 manually written
multi-hop questions following HotpotQA's schema. These should test:
- Bridge questions: "Who is the director of the film that stars X?"
- Comparison questions: "Which paper was published first, A or B?"
Use content from the documents already in data/indices/ where possible.

### 2-3: Run HotpotQA evaluation

Add a dedicated evaluation mode to `benchmarks/run_real_evaluation.py`:

  python benchmarks/run_real_evaluation.py \
    --dataset hotpotqa \
    --systems bm25,flatrag,raptor,treerag_beam \
    --output data/benchmark_reports/hotpotqa_results.json

Report: For HotpotQA multi-hop questions specifically, TreeRAG should show
the largest improvement over single-document methods (hypothesis to test).

### 2-4: Sub-question decomposition for multi-hop (optional but impactful)

If TreeRAG-Beam does NOT outperform RAPTOR on HotpotQA, add a sub-question
decomposition step to `src/core/reasoner.py`:

```python
def _decompose_multihop_query(query: str) -> list[str]:
    """
    If query contains comparison/bridge indicators ("compare", "both",
    "difference", "which", "who"), use Gemini to decompose into
    2 sub-questions. Each sub-question is answered independently,
    then results are merged. Falls back to returning [query] unchanged.
    """
```

This gives TreeRAG an explicit multi-hop advantage over RAPTOR.

---

## PHASE 3 — Medical Domain Specialization

The corresponding author is from Biomedical Engineering. A medical domain
contribution strengthens the ACM submission's domain impact.

### 3-1: Curate medical document set

Create `benchmarks/datasets/medical_corpus.py`:

```python
OPEN_MEDICAL_SOURCES = [
    # PubMed Central Open Access - direct PDF download
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8460250/pdf/",  # example
    # WHO guidelines (public domain)
    "https://apps.who.int/iris/bitstream/handle/10665/272596/9789241565653-eng.pdf",
    # CDC clinical guidelines
]

def prepare_medical_corpus() -> list[str]:
    """
    1. Try to download PDFs from OPEN_MEDICAL_SOURCES into data/raw/medical/.
    2. If downloads are blocked, use existing medical PDFs already in data/raw/:
       - 생체의공학개론#10.pdf
       - 생체의공학개론#11.pdf
       - 생체의공학개론_보고서.pdf
       - s41598-026-41649-2_reference.pdf (Nature Scientific Reports)
    3. Return list of paths to available medical PDFs.
    """
```

### 3-2: Medical Q&A generation

Extend `benchmarks/datasets/auto_qa_generator.py` to support a medical mode:

```python
MEDICAL_QA_PROMPT = """
You are a medical QA dataset creator for evaluating clinical document RAG.
Generate {n} questions that a clinician would realistically ask when
consulting this document.

Document tree:
{tree_json}

Question types to generate:
- clinical_fact (4): Specific clinical values, dosages, criteria
  e.g. "What is the recommended dosage of X for condition Y?"
- procedure (3): Step-by-step clinical procedures
  e.g. "What are the steps for performing X?"
- comparison (2): Comparing conditions/treatments
  e.g. "What distinguishes condition A from condition B?"
- safety (1): Contraindications, warnings, side effects
  e.g. "What are the contraindications for X?"

For each question:
- question: the clinical question
- expected_sections: relevant node IDs
- expected_answer_hint: expected answer in 1-2 sentences
- difficulty: "easy" | "medium" | "hard"
- category: one of the four types above
- clinical_relevance: why a clinician would ask this

JSON only: {"questions": [...]}
"""
```

Generate a `benchmarks/datasets/medical_benchmark.json` with 40+ questions
from the medical documents.

### 3-3: Medical-specific evaluation metrics

Add to `benchmarks/metrics/text_similarity.py`:

```python
def medical_entity_recall(hypothesis: str, reference: str) -> float:
    """
    Measures recall of medical entities (drug names, dosages, anatomy terms,
    condition names) in the hypothesis vs reference.

    Use a simple keyword list approach:
    - Load a medical term list (create a small one with ~200 common terms)
    - Count how many reference medical terms appear in hypothesis
    - Return count_matched / count_in_reference

    Offline, no external APIs needed.
    """

MEDICAL_TERMS = [
    # Anatomy
    "cardiac", "pulmonary", "hepatic", "renal", "cerebral", "vascular",
    "myocardial", "neural", "skeletal", "endocrine",
    # Common conditions
    "hypertension", "diabetes", "myocardial infarction", "stroke", "sepsis",
    "pneumonia", "arrhythmia", "fibrillation", "thrombosis", "embolism",
    # Measurements
    "mmhg", "bpm", "ml/min", "mg/dl", "mcg/kg", "iu/l",
    # Biomedical engineering specific
    "impedance", "electrode", "biosignal", "prosthetic", "scaffold",
    "biocompatibility", "in vitro", "in vivo", "signal-to-noise",
    # ... add more relevant to the actual documents
]
```

### 3-4: Run medical domain evaluation

  python benchmarks/run_real_evaluation.py \
    --dataset medical \
    --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam \
    --output data/benchmark_reports/medical_results.json \
    --domain medical

Hypothesis: TreeRAG's page-level citations and structure preservation are
especially valuable in medical documents where exact source traceability
is clinically required. Show this in the Discussion section.

---

## PHASE 4 — Human Evaluation Framework

ACM reviewers expect human evaluation for RAG system papers.
This phase creates the annotation infrastructure.

### 4-1: Annotation schema

Create `benchmarks/human_eval/annotation_schema.py`:

```python
ANNOTATION_DIMENSIONS = {
    "faithfulness": {
        "description": "Is every factual claim in the answer supported by the source document?",
        "scale": [1, 2, 3, 4, 5],
        "anchors": {
            1: "Answer contains fabricated information not in source",
            3: "Answer mostly accurate with minor unsupported claims",
            5: "Every claim is directly traceable to source document"
        }
    },
    "relevance": {
        "description": "Does the answer directly and completely address the question?",
        "scale": [1, 2, 3, 4, 5],
        "anchors": {
            1: "Answer is off-topic or misses the question entirely",
            3: "Answer partially addresses the question",
            5: "Answer directly and completely addresses the question"
        }
    },
    "citation_quality": {
        "description": "Are source citations specific, accurate, and useful?",
        "scale": [0, 1, 2],
        "anchors": {
            0: "No citations provided",
            1: "Citations present but vague (document-level only)",
            2: "Citations are page-specific and verifiable [Doc, p.X]"
        }
    }
}

INTER_ANNOTATOR_AGREEMENT_THRESHOLD = 0.6  # Krippendorff's alpha target
```

### 4-2: Annotation task generator

Create `benchmarks/human_eval/generate_annotation_tasks.py`:

```python
def generate_annotation_tasks(
    evaluation_results_path: str,
    n_questions: int = 50,
    systems: list = ["raptor", "flatrag", "treerag_beam"],
    output_path: str = "benchmarks/human_eval/annotation_tasks.csv"
) -> None:
    """
    Selects n_questions from the evaluation results (stratified by difficulty
    and category), then for each question generates one annotation row per
    system (blinded — annotator does not see which system produced the answer).

    Output CSV columns:
    task_id, question_id, question, source_excerpt, answer,
    [blank] faithfulness, [blank] relevance, [blank] citation_quality,
    [blank] notes

    The system_name column is EXCLUDED from the CSV to ensure blinding.
    A separate mapping file annotation_key.json maps task_id → system_name.

    Total rows = n_questions × len(systems) = 150 rows
    Estimated annotation time per annotator: ~3 hours
    """
```

### 4-3: Inter-annotator agreement calculator

Create `benchmarks/human_eval/compute_agreement.py`:

```python
def krippendorff_alpha(annotations: dict) -> float:
    """
    Computes Krippendorff's alpha for ordinal data across annotators.
    annotations: {annotator_id: {task_id: score}}
    """

def compute_system_scores(
    annotations_path: str,
    key_path: str
) -> dict:
    """
    After annotation is complete:
    1. Load annotations CSV and annotation_key.json
    2. Compute mean score per system per dimension
    3. Compute Krippendorff's alpha across annotators
    4. Run Wilcoxon signed-rank test between TreeRAG-Beam and each other system
    5. Return and print a formatted table ready for the paper
    """
```

### 4-4: Annotation instructions document

Create `benchmarks/human_eval/ANNOTATION_GUIDE.md` in Korean
(for recruiting student annotators from your lab):

Include:
- Purpose of the annotation (RAG system comparison)
- Explanation of each dimension with concrete examples
- Examples of score 1, 3, and 5 for each dimension
- What to do when unsure
- Estimated time commitment
- Privacy/data handling notes

---

## PHASE 5 — Paper-Ready Output Generation

### 5-1: Updated comparison table (Table 1)

Update `scripts/generate_paper_tables.py` to output the full ACM-level Table 1:

```latex
\begin{table}[t]
\centering
\caption{Main Results on Full Benchmark and HotpotQA Multi-Hop Subset}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{System} & \multicolumn{3}{c}{Full Benchmark} & \multicolumn{2}{c}{HotpotQA} & \\
\cmidrule(lr){2-4} \cmidrule(lr){5-6}
& ROUGE-L & BERTSc. & LLM-J & ROUGE-L & LLM-J & Lat.(s) \\
\midrule
BM25 & & & & & & \\
Dense Retrieval & & & & & & \\
FlatRAG & & & & & & \\
RAPTOR & & & & & & \\
TreeRAG-DFS & & & & & & \\
\textbf{TreeRAG-Beam} & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

Fill all cells with real numbers from the evaluation results.
Bold the best value in each column. Add a \dag footnote for p < 0.05.

### 5-2: Medical domain table (Table 2 for medical-focused version)

```latex
\begin{table}[t]
\centering
\caption{Medical Domain Results — ROUGE-L, Entity Recall, Citation Quality}
...
```

### 5-3: Generate all figures

Update `scripts/plot_results.py` to produce these 4 figures:

Figure 1 — Architecture diagram (already exists in README, recreate as PDF)
Figure 2 — Main results bar chart: all systems × ROUGE-L + BERTScore
Figure 3 — Multi-hop performance: Full benchmark vs HotpotQA subset
            (shows TreeRAG advantage grows on multi-hop questions)
Figure 4 — Context reduction curve: accuracy vs context size for all systems
            (shows TreeRAG Pareto-dominates others)

All figures: DPI=300, PDF format, seaborn-white style, colorblind-safe palette.

### 5-4: Contribution summary for Introduction

After all experiments are run and tables are generated, print a
"contribution summary" to stdout that the author can paste into the
Introduction section:

```
=== CONTRIBUTION SUMMARY (paste into Introduction) ===

This paper makes the following contributions:
1. TreeRAG system: [describe main system]
2. Empirical result: TreeRAG-Beam achieves ROUGE-L of X.XX, outperforming
   RAPTOR by +X.X% and Dense Retrieval by +X.X% on the full benchmark.
3. Multi-hop advantage: On HotpotQA multi-hop questions, TreeRAG-Beam
   outperforms RAPTOR by +X.X% (p=0.0XX), demonstrating that structure
   preservation benefits multi-hop reasoning.
4. Medical domain: TreeRAG achieves X.XX entity recall on medical documents,
   with 100% page-level citation availability vs 0% for RAPTOR.
5. Open source: Code and datasets at [GitHub URL].
=======================================================
```

---

## Completion Criteria

After all phases, verify:

1. pytest -q --tb=short passes with MORE tests than before you started.
2. python benchmarks/run_real_evaluation.py --dataset full_benchmark.json
   --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam
   prints a complete comparison table with RAPTOR included.
3. python benchmarks/run_real_evaluation.py --dataset hotpotqa
   --systems bm25,flatrag,raptor,treerag_beam
   prints HotpotQA results.
4. python benchmarks/run_real_evaluation.py --dataset medical
   --systems bm25,flatrag,raptor,treerag_beam
   prints medical domain results.
5. benchmarks/human_eval/annotation_tasks.csv exists with 150 rows.
6. benchmarks/human_eval/ANNOTATION_GUIDE.md exists in Korean.
7. python scripts/generate_paper_tables.py outputs three LaTeX tables.
8. python scripts/plot_results.py generates four PDF figures.
9. The "contribution summary" is printed with real numbers filled in.

Finally, print a file diff summary showing every new file created and
every existing file modified, so the author knows exactly what changed.
```
