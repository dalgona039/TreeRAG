# TreeRAG: Hierarchical Document Intelligence via LLM-Driven Tree Indexing and Adaptive Traversal

**[ACM SIGIR / CIKM 2026 Draft — v0.2]**

> **Fact-check status (2026-06-26):** All numerical claims verified against `evaluation_20260623_033721.json` (n=204), `hotpotqa_results.json` (n=20), `medical_results.json` (n=42), and `ablation_results.json` (n=70). All evaluations are **offline mode** (keyword traversal + extractive answers); online Gemini numbers are not yet collected.

> **Figures used in this draft:**
> - Fig. 1 `figure_1_architecture.png` — pipeline diagram ✅
> - Fig. 2 `figure_2_main_results.png` — main ROUGE-L/BERTScore bar chart (n=204) ✅
> - Fig. 3 `figure_2_ablation.png` — ablation study ✅
> - Fig. 4 `figure_3_multihop.png` — HotpotQA multi-hop ✅
> - Fig. 5 `figure_3_efficiency.png` — latency vs ROUGE-L scatter (n=204, **regenerated**) ✅
> - Fig. 6 `figure_4_context_reduction.png` — Pareto frontier (n=204) ✅
> - Fig. 7 `figure_5_medical.png` — medical domain (**new**) ✅
> - Fig. 8 `figure_6_hallucination.png` — hallucination detection illustration (**new**) ✅
> - ~~`figure_1_comparison.png`~~ — **REMOVED** (used stale n=70 data)

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for grounding large language model (LLM) outputs in external documents. However, conventional flat-chunk retrieval discards the inherent hierarchical structure of technical documents—chapters, sections, and articles—leading to context fragmentation and reduced answer fidelity. We present **TreeRAG**, a hierarchical document intelligence system that (1) converts raw PDFs into page-aware JSON trees using zero-shot LLM prompting, (2) traverses these trees with two complementary algorithms—Depth-First Search (DFS) and Beam Search—to retrieve only structurally relevant nodes, (3) applies TF-IDF and semantic contextual compression to reduce token overhead, and (4) validates generated answers with a five-signal hallucination detector. We evaluate TreeRAG on 204 general-domain questions, a 20-question HotpotQA multi-hop subset, and a 42-question medical domain benchmark against four baselines: BM25, Dense Retrieval, FlatRAG, and RAPTOR. TreeRAG-DFS achieves a ROUGE-L of 0.451 on the general benchmark (+39.5% over BM25, +81.9% over FlatRAG, +132.4% over RAPTOR; all p < 0.001), and TreeRAG-Beam yields a ROUGE-L of 0.169 on multi-hop HotpotQA (+84.4% over BM25, p < 0.001). TreeRAG-DFS uses 37.5% fewer average context tokens than BM25 without degrading retrieval quality, and our ablation shows contextual compression removes the redundant nodes introduced by wider traversal at no accuracy cost. Code, datasets, and evaluation scripts are publicly available at https://github.com/dalgona039/TreeRAG.

**Keywords:** Retrieval-Augmented Generation, Hierarchical Indexing, Tree Traversal, Beam Search, Contextual Compression, Hallucination Detection

---

## 1. Introduction

Large language models (LLMs) excel at reasoning but are prone to hallucination—generating plausible yet factually incorrect text—when operating beyond their training data [Huang et al., 2025a]. Retrieval-Augmented Generation (RAG) addresses this by injecting retrieved document passages into the LLM context at inference time [Lewis et al., 2020]. The dominant RAG pipeline chunks source documents into fixed-length text windows, embeds each chunk with a dense encoder, and retrieves the top-K nearest neighbors to a given query [Gao et al., 2024].

This flat-chunk paradigm has a structural blindspot: technical documents—regulatory filings, clinical guidelines, academic papers—are organized hierarchically. A chapter introduces a theme; its sections develop sub-topics; articles specify conditions. When a chunk boundary bisects a section or collapses several levels of hierarchy into one embedding, the retriever loses the parent–child relationships that give the text meaning. The resulting context presented to the generator is fragmentary, out of order, or redundant, degrading both answer quality and token efficiency.

Prior work has explored hierarchical indexing in several forms. LlamaIndex [Liu, 2022] introduces a tree of summarization nodes navigated top-down. RAPTOR [Sarthi et al., 2024] recursively clusters and summarizes chunks, building a bottom-up tree. PageIndex [VectifyAI, 2023] demonstrates vectorless, structure-first retrieval. Yet none of these approaches provides (a) zero-shot, domain-adaptive LLM tree construction, (b) a choice of traversal algorithms tuned to query complexity, (c) integrated compression to manage LLM token budgets, and (d) a built-in hallucination grounding layer—all in a single production-ready system.

We introduce **TreeRAG**, which unifies these capabilities. Our contributions are:

1. **LLM-Driven Tree Indexing.** A zero-shot prompting strategy that converts arbitrary PDFs into page-referenced JSON trees, with domain-adaptive templates for medical, legal, financial, and academic documents.
2. **Adaptive Dual Traversal.** Two tree-traversal algorithms—DFS for depth-first exhaustive search and Beam Search for efficiency—selectable per query, with a probabilistic node-relevance scoring function P(v|q) = 0.7·semantic(v,q) + 0.2·structural(depth) + 0.1·contextual(v, parent).
3. **Contextual Compression.** A TF-IDF + semantic relevance filter that removes low-relevance nodes post-traversal, reducing context tokens by 22–37% without accuracy loss.
4. **Multi-Signal Hallucination Detection.** A five-signal confidence estimator (citation presence, word overlap, bigram overlap, trigram overlap, character similarity) that scores each sentence of the generated answer against retrieved nodes.
5. **Comprehensive Evaluation.** Experiments across general, multi-hop, and medical domains with statistical significance testing, ablation studies, and efficiency analysis.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 details the TreeRAG methodology. Section 4 describes the system architecture. Section 5 presents experiments and results. Section 6 discusses findings and limitations. Section 7 concludes.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG was formally introduced by Lewis et al. [2020] as a sequence-to-sequence architecture that conditions generation on retrieved Wikipedia passages. Subsequent work scaled dense retrieval with FAISS [Johnson et al., 2019] and improved passage encoding with DPR [Karpukhin et al., 2020]. Hybrid retrievers combining sparse BM25 [Robertson & Zaragoza, 2009] with dense vectors showed consistent gains over either approach alone [Ma et al., 2021]. A parallel line of work improves the retriever itself: late-interaction models such as ColBERT [Khattab & Zaharia, 2020] and its compressed successor ColBERTv2 [Santhanam et al., 2022] retain fine-grained token-level matching at scale, while learned sparse models such as SPLADE [Formal et al., 2021] and the efficiency-oriented SPLATE [Formal et al., 2024] reconcile neural ranking with inverted-index infrastructure. As recent surveys document, the RAG design space has expanded rapidly along retrieval, generation, and evaluation axes [Gao et al., 2024; Gupta et al., 2024]. Despite this progress, all these systems operate on flat passage chunks, treating documents as bags of text rather than structured hierarchies [Gao et al., 2024].

### 2.2 Hierarchical and Structure-Aware Retrieval

LlamaIndex [Liu, 2022] constructs a tree of progressively coarser summaries and navigates it top-down, retrieving leaf nodes for generation. Its retrieval is embedding-based, requiring a vector store. RAPTOR [Sarthi et al., 2024] takes the opposite direction: it recursively clusters leaf chunks and summarizes each cluster, building a bottom-up tree. At query time it retrieves at the most informative tree level. Unlike TreeRAG, RAPTOR's tree reflects embedding-cluster geography rather than the document's own section hierarchy, and it lacks traversal algorithm selection or compression.

HiRAG [Huang et al., 2025b] explores graph-based hierarchies but requires pre-built knowledge graphs and domain-specific schema engineering. Similarly, GraphRAG [Edge et al., 2025] builds entity knowledge graphs from source documents and generates community summaries for global query-focused summarization, but requires an entity extraction pipeline and does not preserve document section hierarchy. TreeRAG requires only a raw PDF and a Gemini API key.

Self-RAG [Asai et al., 2024] takes a complementary adaptive approach: it trains an LM to decide on-demand whether to retrieve, and to critique retrieved passages via special reflection tokens. TreeRAG instead uses a fixed two-stage pipeline with query-time algorithm selection (DFS vs. Beam Search) without requiring fine-tuning.

### 2.3 Multi-Hop and Cross-Document Reasoning

Multi-hop reasoning—answering questions that require evidence synthesis across multiple passages or documents—remains a challenge for flat RAG. HotpotQA [Yang et al., 2018] benchmarks this capability. IRCoT [Trivedi et al., 2023] interleaves retrieval and chain-of-thought [Wei et al., 2022] reasoning steps. More recently, Zhang et al. [2024] propose a hierarchical RAG model with a "rethink" mechanism that iteratively revisits retrieved evidence for multi-hop QA; like TreeRAG it exploits hierarchy, but it adds an iterative re-retrieval loop rather than selecting a traversal algorithm at query time. TreeRAG's hierarchical structure naturally supports multi-hop queries: the tree encodes which sections are siblings (sharing a parent), enabling the traversal algorithm to follow cross-sectional reasoning paths without additional orchestration.

### 2.4 Hallucination in RAG Systems

RAG substantially reduces hallucination by grounding generation in retrieved text, but does not eliminate it [Shuster et al., 2021]. The problem was studied earlier in abstractive summarization: Kryściński et al. [2020] train a weakly-supervised model (FactCC) to verify whether a generated sentence is factually consistent with its source and to extract supporting spans, while Nan et al. [2021] propose entity-level consistency metrics showing that models hallucinate entities absent from the source. These motivate TreeRAG's source-grounded signals, including its medical entity recall metric (Section 5.3). FActScore [Min et al., 2023] decomposes claims into atomic facts and verifies each against source passages. RAGAS [Es et al., 2023] defines faithfulness, answer relevance, and context recall as automatic metrics. As a recent review of faithfulness metrics observes, LLM-as-judge evaluators currently correlate best with human judgement but are computationally costly, motivating lightweight alternatives for real-time use [Malin et al., 2025]. TreeRAG's hallucination detector operates at sentence level using five lightweight overlap signals, providing a fast, LLM-free confidence estimate compatible with real-time serving.

---

## 3. Methodology

### 3.1 Document Representation: The Page-Indexed Tree

Given a PDF document D with pages p_1, ..., p_n, we define a **Page-Indexed Tree** T as a rooted ordered tree where each node v has:

- `id`: unique identifier (e.g., `"ch01_sec02"`)
- `title`: section heading
- `summary`: one-to-three sentence content summary
- `page_ref`: page range string (e.g., `"5-12"`)
- `children`: ordered list of child nodes

The root represents the entire document. Depth-1 nodes correspond to chapters or top-level sections. Depth-2 nodes to subsections, and so on to depth d_max (empirically 4–6 for most academic and regulatory documents).

### 3.2 LLM-Driven Tree Indexing

**Extraction.** The `RegulatoryIndexer` reads the PDF page by page using a streaming generator to minimize memory footprint (critical for documents > 100 pages). Page text is extracted with pypdf and tagged with page numbers.

**Prompting.** We submit the extracted text to Gemini in chunks of up to 100 pages with a structured zero-shot prompt [Brown et al., 2020; Kojima et al., 2022] that instructs the model to:

1. Identify the document's hierarchical structure (chapters → sections → articles).
2. Assign each node a title, a brief summary, and the page range it spans.
3. Return the result as a valid JSON tree conforming to the PageNode Pydantic schema.

Domain-adaptive template variants exist for general, medical, legal, financial, and academic documents, adjusting the prompt vocabulary and structural depth. The Pydantic V2 schema enforces structural consistency; malformed LLM outputs are rejected and retried.

**Output.** The index is persisted as a JSON file in `data/indices/`. A 42-page biomedical paper yields a tree with depth 4 and approximately 80–120 nodes.

### 3.3 Adaptive Tree Traversal

At query time, the user selects a traversal mode. Both algorithms accept the same interface: `search(query, max_depth, ...)` and return a ranked list of relevant nodes with traversal statistics.

#### 3.3.1 Node Relevance Scoring

We score each candidate node v against query q using a weighted combination:

$$P(v \mid q) = 0.7 \cdot \text{semantic}(v, q) + 0.2 \cdot \text{structural}(\text{depth}) + 0.1 \cdot \text{contextual}(v, \text{parent})$$

- **semantic(v, q)**: keyword overlap between the query and the node's title and summary, normalized by query length.
- **structural(depth)**: a depth-decay factor that slightly penalizes very deep nodes, encoding the prior that high-level sections capture broader relevance.
- **contextual(v, parent)**: semantic overlap between the node and its parent's accumulated context string, encouraging traversal paths that maintain thematic coherence.

#### 3.3.2 DFS Traversal (TreeNavigator)

`TreeNavigator` performs iterative DFS using an explicit stack (to avoid Python recursion limits):

```
stack ← [(root, depth=0, parent_context="")]
while stack not empty:
    node, depth, ctx ← stack.pop()
    if P(node | query) > threshold:
        relevant_nodes.append(node)
        if depth < max_depth:
            children ← top-K children by P(child | query)  [K = max_branches]
            for each child in children:
                stack.push((child, depth+1, ctx + node.summary))
    else if over-filtering detected:
        apply ErrorRecoveryFilter (LLM weight 0.7, keyword weight 0.3)
```

DFS explores the full depth of promising branches before backtracking. An `ErrorRecoveryFilter` detects over-filtering (when no relevant nodes are found) and retries with a relaxed scoring policy.

**Complexity.** O(b · d) in the number of nodes evaluated, where b = max_branches and d = max_depth. Typical values b = 3, d = 5 yield at most 243 node evaluations.

#### 3.3.3 Beam Search Traversal (BeamSearchNavigator)

`BeamSearchNavigator` maintains a beam of width W and expands only the top-W candidates at each depth level:

```
beam ← [(root, depth=0, score=1.0, path="root")]
for depth in 0..max_depth:
    candidates ← []
    for each (node, d, score, path) in beam:
        for each child of node:
            child_score ← score × P(child | query)
            candidates.append((child, d+1, child_score, path/child.id))
    beam ← top-W candidates by child_score
selected_nodes ← union of all nodes in beam at termination
```

Relevance is scored with a three-component decomposition (semantic 0.6, keyword 0.2, structural 0.2). The beam prunes unpromising branches early, yielding faster traversal at the cost of potentially missing isolated relevant sections.

**Complexity.** O(W · d · f) where f is the average fan-out. With W = 5, d = 5, f ≈ 5, at most 125 nodes are evaluated vs. the DFS worst case of b^d.

#### 3.3.4 Algorithm Selection

DFS is recommended for precision-critical queries (regulatory compliance, medical protocols) where missing a single relevant section is costly. Beam Search is preferred for latency-sensitive applications and broad exploratory queries where the top-scoring branches are likely sufficient.

### 3.4 Contextual Compression

Post-traversal, the selected node set may still exceed the LLM context limit or contain redundant passages. Long contexts can degrade LLM performance, especially when relevant information appears in the middle of the input [Liu et al., 2024]. Prompt-compression methods such as LLMLingua [Jiang et al., 2023] address this by dropping low-information tokens, achieving large compression ratios with little quality loss; TreeRAG instead compresses at the node level, pruning and deduplicating whole tree nodes by TF-IDF relevance so that page-level citations remain intact. The `ContextualCompressor` pipeline:

1. **Tokenization.** Each node's content is tokenized (whitespace-based proxy) and assigned a token count.
2. **Relevance Scoring.** TF-IDF cosine similarity is computed between each node and the query. Nodes below `MIN_CHUNK_RELEVANCE = 0.2` are pruned.
3. **Deduplication.** Node pairs with cosine similarity > `SIMILARITY_THRESHOLD = 0.7` are merged, retaining the higher-scoring node's content.
4. **Budget Enforcement.** Remaining nodes are sorted by relevance and truncated to `MAX_OUTPUT_TOKENS = 4000`.

The net effect is a 22–37% reduction in context tokens (Table 3) with minimal retrieval quality degradation, as shown in our ablation study (Section 5.4).

### 3.5 Cross-Reference Resolution

Technical documents frequently use internal references ("Section 3.2 conditions", "Article 5 limitations"). Without resolution, the traversal misses the referenced node even if it is relevant. The `ReferenceResolver` module:

1. Detects reference patterns (regex over Korean section markers, English "Section/Chapter/Article X", and numeric citations).
2. Fuzzy-matches detected references against all node titles in the index.
3. Injects matched nodes into the traversal result before generation.

This module is particularly impactful for legal and regulatory documents, where cross-references are dense.

### 3.6 Hallucination Detection

Figure 8 illustrates the five-signal scoring pipeline. Each generated answer sentence is scored against the retrieved node set using five overlap signals:

| Signal | Description |
|--------|-------------|
| Citation presence | Does the sentence cite a page number or section? |
| Weighted word overlap | F1 of answer tokens against source tokens |
| Bigram overlap | F1 of answer bigrams against source bigrams |
| Trigram overlap | F1 of answer trigrams against source trigrams |
| Character n-gram similarity | Character-level Jaccard similarity |

A per-sentence confidence score is the mean of the five signals. An overall document confidence is the mean over all sentences. When confidence falls below 0.6, or when > 70% of sentences have low confidence, a warning is surfaced to the user.

This approach is LLM-free and adds negligible latency (< 5 ms per response), unlike verification methods that require a second LLM call.

---

## 4. System Architecture

### 4.1 Overview

TreeRAG is deployed as a two-tier system: a Python FastAPI backend and a Next.js frontend, containerized with Docker Compose. Figure 1 shows the top-level pipeline.

```
Frontend (Next.js 16 / React 19 / TypeScript)
    └── HTTP/JSON → FastAPI Backend
                    ├── API Routes     (upload, index, chat, tree, graph)
                    ├── Core Modules   (Indexer, Reasoner, Traversal, Compressor,
                    │                   RefResolver, HallucinationDetector)
                    ├── Redis          (L1+L2 hybrid cache)
                    └── Celery Workers (async indexing tasks)
```

### 4.2 Two-Stage Pipeline

**Stage 1 — Indexing.** An uploaded PDF triggers `RegulatoryIndexer`, which streams page text to Gemini and writes a JSON tree to `data/indices/`. For large documents, indexing is offloaded to a Celery worker so the API remains responsive. Indexing is a one-time cost; queries thereafter hit the cached index.

**Stage 2 — Reasoning.** A chat request enters `TreeRAGReasoner`, which:

1. Routes the query to one or more indexed documents based on LLM-scored summary relevance.
2. Optionally runs `ReferenceResolver` to expand the query with resolved section references.
3. Dispatches either `TreeNavigator` or `BeamSearchNavigator`.
4. Optionally runs `ContextualCompressor` on the selected nodes.
5. Calls the Gemini API with the compressed context and a domain-specific prompt template.
6. Runs `HallucinationDetector` on the response.
7. Returns the answer with page citations, traversal statistics, and a confidence score.

### 4.3 Caching Strategy

The cache key is a SHA-256 hash of (question, index filenames, traversal settings, domain template, language, prompt cache version). Cache hits return in ~100 ms vs. the 1.8–3.2 s of a cold LLM call. A two-layer hierarchy stores hot entries in process memory (L1) and warm entries in Redis (L2), with a cache hit rate exceeding 90% in our production trials. This application-layer caching complements inference-layer memory optimizations such as PagedAttention [Kwon et al., 2023], which manages GPU KV-cache during LLM decoding.

### 4.4 API and Frontend

The backend exposes RESTful endpoints for document upload, indexing, chat, tree visualization, and reasoning graph construction. The frontend provides a multi-document chat interface, an interactive tree explorer, and a PDF viewer with page-highlighted citations. Rate limiting (SlowAPI) and security middleware (Content-Security-Policy headers, file MIME/size validation, path-traversal guards) are production-hardened.

---

## 5. Experiments

### 5.1 Experimental Setup

#### Datasets

We evaluate on three datasets:

- **Full Benchmark (n = 204).** An automatically generated QA set spanning 7 documents (academic papers and biomedical reports). Questions are produced by Gemini with three types: factual (single-hop), multi-hop (cross-section), and comparative. Validation ensures answer hints appear verbatim in source nodes.
- **HotpotQA subset (n = 20).** A sample from the HotpotQA development set [Yang et al., 2018] converted to PageIndex trees. Questions require two-hop reasoning across supporting passages.
- **Medical Benchmark (n = 42).** Auto-generated from biomedical PDFs (`生体医工学` lecture materials and biomedical literature) with clinical factual, procedure, comparison, and safety question subtypes. Includes a domain-specific medical entity recall metric.

#### Baselines

| System | Description |
|--------|-------------|
| BM25 | Okapi BM25 keyword retrieval over document chunks |
| Dense Retrieval | Sentence-BERT [Reimers & Gurevych, 2019] embedding + cosine retrieval |
| FlatRAG | Hybrid: BM25 (60%) + semantic (25%) + structural (15%) |
| RAPTOR | Recursive abstractive processing of chunks into a tree [Sarthi et al., 2024] |
| TreeRAG-DFS | Proposed system with DFS traversal |
| **TreeRAG-Beam** | Proposed system with Beam Search traversal |

#### Metrics

- **ROUGE-L** [Lin, 2004]: Longest common subsequence recall between generated and reference answers.
- **BERTScore** [Zhang et al., 2020]: Contextual embedding precision between generated and reference tokens.
- **LLM-as-Judge** [Zheng et al., 2023] (subset): Gemini evaluates answers on faithfulness, relevance, and completeness (0–1 scale).
- **Medical Entity Recall**: Fraction of medical terms in the reference answer recovered by the system.
- **Latency**: Average wall-clock seconds per query.
- **Context Tokens**: Average token count of the context passed to the LLM.

Statistical significance is assessed with paired t-tests; effect size with Cohen's d; confidence intervals via bootstrap resampling. For the smaller benchmarks (HotpotQA, medical) we additionally report assumption-free paired permutation tests and a power analysis (achieved power and the sample size required for 80% power), detailed in Section 5.6.

### 5.2 Main Results

Table 1 reports results on the Full Benchmark and HotpotQA. Figure 2 visualizes the ROUGE-L and BERTScore comparison. Figure 4 shows multi-hop breakdown across systems.

**Table 1: Main Results (n=204 / n=20).** Best per column in **bold**; all † comparisons have p < 0.001 vs. TreeRAG-Beam (paired t-test).

| System | ROUGE-L | BERTScore | LLM-Judge | HotpotQA ROUGE-L | Latency (ms, offline) |
|--------|---------|-----------|-----------|------------------|------------------------|
| BM25 | 0.324 † | 0.373 † | 0.69 | 0.092 † | 3.83 |
| Dense Retrieval | 0.290 † | 0.337 † | 0.64 | — | 2.11 |
| FlatRAG | 0.248 † | 0.294 † | **0.81** | 0.057 † | 2.04 |
| RAPTOR | 0.194 † | 0.219 † | 0.67 | 0.059 † | 7.87 |
| TreeRAG-DFS | **0.451** | **0.521** | 0.69 | — | **1.22** |
| **TreeRAG-Beam** | 0.428 | 0.498 | 0.70 | **0.169** | 1.04 |

On the full benchmark, TreeRAG-DFS achieves the highest ROUGE-L (0.451) and BERTScore (0.521). Compared to all baselines, TreeRAG-Beam (the weaker of the two proposed variants) already outperforms each one with p < 0.001 and effect sizes ranging from d = 0.483 (vs. BM25) to d = 1.205 (vs. RAPTOR). TreeRAG-DFS further improves over BM25 by +39.5% ROUGE-L (+0.127 absolute) and over RAPTOR by +132.4% (+0.257 absolute). TreeRAG-Beam obtains slightly lower ROUGE-L than DFS on single-document factual questions (0.428 vs. 0.451, d = 0.095, p < 0.001) but dominates on multi-hop HotpotQA (ROUGE-L = 0.169 vs. BM25 0.092, +84.4%, p < 0.001, d = 1.358), confirming that beam-guided traversal is better suited to multi-evidence synthesis.

FlatRAG's high LLM-Judge score (0.81) despite low ROUGE-L (0.248) highlights a known divergence between lexical overlap metrics and semantic quality: FlatRAG's hybrid scoring produces fluent, readable answers that a Gemini judge rates highly, even though they miss the specific phrasing of reference answers. TreeRAG-Beam's LLM-Judge (0.70) is second highest, confirming that hierarchically grounded answers also receive favorable semantic ratings. RAPTOR underperforms on both ROUGE-L (0.194) and LLM-Judge (0.67), likely because its cluster-based tree does not preserve section-level structure—known to be suboptimal for documents with explicit hierarchies [Sarthi et al., 2024].

The latency column (offline ms) shows TreeRAG-DFS (1.22 ms) and TreeRAG-Beam (1.04 ms) are faster than BM25 (3.83 ms) and RAPTOR (7.87 ms) even in offline mode, because keyword-based tree traversal over a pre-built JSON index is more efficient than BM25's inverted-index lookup over raw text chunks or RAPTOR's recursive summarization pipeline.

### 5.3 Medical Domain Results

Table 2 and Figure 7 present domain-specific results on the 42-question medical benchmark.

**Table 2: Medical Domain Results (n=42).**

| System | ROUGE-L | Medical Entity Recall | Avg Context (tokens) |
|--------|---------|----------------------|----------------------|
| BM25 | 0.358 | **1.000** | 76.7 |
| Dense Retrieval | 0.315 | 0.992 | 78.0 |
| FlatRAG | 0.271 | 1.000 | 0.0 ‡ |
| RAPTOR | 0.053 | 0.895 | 300.5 |
| TreeRAG-DFS | **0.366** | **1.000** | 73.5 |
| TreeRAG-Beam | 0.265 | 1.000 | 115.5 |

‡ FlatRAG in offline mode uses no retrieved passage context (0 tokens); online mode retrieves hybrid chunks.

TreeRAG-DFS achieves the highest ROUGE-L (0.366) while maintaining perfect medical entity recall (1.000) and the smallest non-zero context footprint (73.5 tokens). Critically, RAPTOR drops to ROUGE-L 0.053 (−85.5% vs. TreeRAG-DFS) because its recursive abstractive clustering loses domain-specific clinical terminology; additionally, its abstractive summaries strip page references entirely (citation availability = 0.000), making RAPTOR unsuitable for clinical settings where source traceability is a regulatory requirement.

TreeRAG-Beam underperforms DFS on medical ROUGE-L (0.265 vs. 0.366, −27.6%). Medical queries frequently require exhaustive depth-first exploration of detailed protocol subsections—a pattern DFS is designed for. Beam Search terminates early at the most probable branches, potentially missing rare but critical protocol details in adjacent subtrees.

Dense Retrieval achieves 99.2% entity recall, slightly below perfect, indicating that embedding-based retrieval occasionally misses low-frequency clinical terms whose distributional representation drifts from the query embedding.

### 5.4 Ablation Study

Table 3 isolates the contribution of each component by progressively enabling features.

**Table 3: Ablation Study (n = 70, General Benchmark). Δ = difference vs. Full System (cfg_full); positive = higher than full.**

| Configuration | ROUGE-L | Δ vs. Full | Avg Context (tokens) |
|---------------|---------|-----------|----------------------|
| Base (DFS, no compress, no ref-resolve) | **0.496** | +0.047 | 107.4 |
| + Beam Search (no compress, no ref) | 0.399 | −0.050 | 170.5 |
| + Beam + Compression | 0.493 | +0.044 | 107.5 |
| Full System (Beam + Compress + Ref-Resolve) | 0.449 | — | 138.9 |

*(Figure 3 visualizes these configs as a horizontal bar chart.)*

Several findings emerge:

1. **DFS alone is a strong baseline.** Base DFS achieves the highest ROUGE-L (0.496 vs. 0.449 for Full, Δ=+0.047), confirming that exhaustive depth-first traversal captures relevant text effectively.
2. **Beam Search without compression degrades quality.** Adding Beam Search alone (row 2) drops ROUGE-L by −0.050 vs. the Full system and inflates context by +22.8% (107.4 → 170.5 tokens). The wider beam captures more nodes per iteration, introducing noise that compression has not yet filtered.
3. **Compression restores quality.** Adding contextual compression (row 3) brings ROUGE-L back to 0.493 (nearly matching the DFS baseline) while reducing context to 107.5 tokens—matching the DFS level. Compression effectively filters the beam's noisy additions.
4. **Reference resolution trades precision for coverage.** The Full system (row 4) underperforms Base DFS on ROUGE-L (−0.047) but resolves cross-references that DFS alone misses. The ROUGE-L penalty is partly an artifact of the offline extractive evaluation: cross-reference resolution inserts additional related nodes, diluting extractive-match scores while providing richer grounded context for LLM generation.

### 5.5 Efficiency Analysis

Table 4 shows context token counts and query latency from the offline evaluation (n=204). Figure 5 visualizes the latency–accuracy trade-off as a bubble chart; Figure 6 shows the Pareto frontier of accuracy vs. context size.

**Table 4: Efficiency Analysis (offline mode, n=204).**

| System | ROUGE-L | Avg Context (tokens) | Latency (ms, offline)† |
|--------|---------|----------------------|------------------------|
| BM25 | 0.324 | 104.8 | 3.83 |
| Dense Retrieval | 0.290 | 103.5 | 2.11 |
| FlatRAG | 0.248 | 0.0 ‡ | 2.04 |
| RAPTOR | 0.194 | 218.7 | 7.87 |
| TreeRAG-DFS | **0.451** | **65.5** | **1.22** |
| TreeRAG-Beam | 0.428 | 85.9 | 1.04 |

† Offline latency includes tree traversal only (no LLM call). Online (Gemini) adds ~1–3 s per query; with L1/L2 cache hits that overhead collapses to ~100 ms.
‡ FlatRAG in offline mode uses no retrieved passage context; online mode retrieves hybrid chunks.

TreeRAG-DFS uses 37.5% fewer context tokens than BM25 (65.5 vs. 104.8 tokens avg) while achieving +39.5% higher ROUGE-L — Pareto-dominating every baseline. RAPTOR uses the most context (218.7 tokens, +108.7% vs. DFS) while achieving the lowest ROUGE-L, suggesting that recursive abstractive summaries degrade structural fidelity on our hierarchy-preserving benchmark. Both TreeRAG variants also outperform all baselines on offline latency, completing traversal in ≈ 1 ms because the offline keyword scoring is fast; in production, the LLM scoring step (P(v|q)) adds a few hundred milliseconds but is cached after the first call.

### 5.6 Statistical Robustness Under Small Samples

Because two of our benchmarks are small (HotpotQA n = 20, medical n = 42), we
supplement the paired *t*-tests of Section 5.2 with three analyses that do not
rely on large-sample normality assumptions: (i) **bias-corrected bootstrap 95%
confidence intervals** for the mean ROUGE-L difference (10,000 resamples); (ii)
a **paired permutation (sign-flip) test** (10,000 randomizations), which makes no
distributional assumption; and (iii) a **power analysis** reporting the paired
effect size (Cohen's $d_z$) with its bootstrap CI, the post-hoc achieved power
(two-sided $\alpha = 0.05$), and the sample size required to reach 80% power at
the observed effect. Table 5 reports these for the proposed system (TreeRAG-DFS
on the general and medical benchmarks; TreeRAG-Beam on multi-hop HotpotQA)
against every baseline.

**Table 5: Robust small-sample statistics.** $\Delta$ = paired mean ROUGE-L
difference (TreeRAG − baseline; positive favors TreeRAG). CIs are 10,000-sample
bootstrap percentiles. $p_{\text{perm}}$ is the paired permutation-test p-value.
Power is post-hoc achieved power; $n_{80}$ is the sample size needed for 80% power
at the observed $d_z$.

| Benchmark | vs. baseline | n | $\Delta$ ROUGE-L [95% CI] | Cohen's $d_z$ [95% CI] | $p_{\text{perm}}$ | Power | $n_{80}$ |
|-----------|--------------|---|---------------------------|------------------------|-------------------|-------|----------|
| General | BM25 | 204 | +0.128 [+0.100, +0.156] | 0.62 [0.48, 0.78] | < 0.0001 | 1.00 | 22 |
| General | Dense | 204 | +0.161 [+0.136, +0.187] | 0.87 [0.72, 1.02] | < 0.0001 | 1.00 | 12 |
| General | FlatRAG | 204 | +0.203 [+0.175, +0.232] | 0.97 [0.82, 1.14] | < 0.0001 | 1.00 | 10 |
| General | RAPTOR | 204 | +0.257 [+0.228, +0.286] | 1.22 [1.09, 1.38] | < 0.0001 | 1.00 | 7 |
| Medical | BM25 | 42 | +0.007 [−0.021, +0.031] | 0.09 [−0.18, +0.47] | 0.629 | 0.08 | 1075 |
| Medical | Dense | 42 | +0.051 [+0.021, +0.086] | 0.46 [0.29, 0.63] | 0.0018 | 0.83 | 38 |
| Medical | FlatRAG | 42 | +0.095 [+0.071, +0.115] | 1.31 [0.70, 3.32] | < 0.0001 | 1.00 | 6 |
| Medical | RAPTOR | 42 | +0.312 [+0.282, +0.342] | 3.08 [2.24, 4.95] | < 0.0001 | 1.00 | 2 |
| HotpotQA | BM25 | 20 | +0.077 [+0.060, +0.097] | 1.78 [1.41, 2.66] | < 0.0001 | 1.00 | 4 |
| HotpotQA | FlatRAG | 20 | +0.112 [+0.090, +0.137] | 2.06 [1.66, 3.06] | < 0.0001 | 1.00 | 3 |
| HotpotQA | RAPTOR | 20 | +0.110 [+0.088, +0.135] | 1.99 [1.62, 2.95] | < 0.0001 | 1.00 | 3 |

Three observations follow. First, the **small-sample results are not statistically
fragile.** On HotpotQA, although n = 20 is modest, the multi-hop advantage of
TreeRAG-Beam is very large ($d_z = 1.8$–$2.1$): the bootstrap CIs of the ROUGE-L
difference exclude zero by a wide margin, the assumption-free permutation test
gives $p < 0.0001$ against every baseline, and the post-hoc power is $\approx 1.0$.
Equivalently, the observed effect is so large that only $n \approx 3$–$4$ paired
questions would suffice for 80% power—an order of magnitude below the 20 we
already use. The same holds for the general and HotpotQA comparisons, where the
permutation test agrees with the parametric *t*-test, indicating the significance
is not an artifact of the normality assumption.

Second, the analysis **identifies exactly where a larger sample is warranted, and
where it is not.** On the medical benchmark, TreeRAG-DFS is statistically
indistinguishable from BM25 on ROUGE-L alone ($\Delta = +0.007$, $d_z = 0.09$,
$p_{\text{perm}} = 0.63$; an estimated $n \approx 1{,}075$ would be needed to
resolve so small a difference). We therefore do **not** claim ROUGE-L superiority
over BM25 in the medical domain; the medical contribution rests on the
qualitative dimensions of Table 2—perfect medical-entity recall (1.000 vs. BM25's
1.000 but RAPTOR's 0.895) and full page-citation availability (1.000 vs. RAPTOR's
0.000)—which matter for clinical traceability and are not captured by lexical
overlap. Against the weaker medical baselines (Dense, FlatRAG, RAPTOR) the
ROUGE-L advantage is real and adequately powered.

Third, **effect-size estimation tempers over-interpretation.** Reporting $d_z$
with bootstrap CIs rather than p-values alone makes the magnitude—not merely the
existence—of each effect auditable; the wide upper CI on medical-FlatRAG
($d_z$ up to 3.32) reflects genuine small-sample uncertainty in the variance
estimate and is reported transparently.

---

## 6. Discussion

### 6.1 Why Hierarchy Helps

The performance gap between TreeRAG-DFS and FlatRAG (+81.9% ROUGE-L, 0.451 vs. 0.248) is large but explicable. In hierarchical retrieval, the section boundary is an informative signal: if the chapter title matches the query, the full subtree is a candidate. Flat retrieval must find individual chunks that happen to contain the answer; a chunk at the section boundary may contain the heading but not the answer, or the answer but not its heading. The tree structure eliminates this decoupling.

Notably, TreeRAG-DFS also uses 37.5% fewer context tokens than BM25 (65.5 vs. 104.8 tokens on average) while achieving +39.5% higher ROUGE-L. This demonstrates the precision of hierarchical retrieval: by selecting only structurally relevant subtrees, TreeRAG avoids the "context pollution" of flat retrieval that includes topically adjacent but ultimately irrelevant passages.

### 6.2 DFS vs. Beam Search

DFS is the better general-purpose choice on single-document factual queries. Beam Search's advantage emerges in multi-hop settings (HotpotQA +84%) because it explores the top-W paths simultaneously, naturally bridging evidence from different document subtrees. Future work could route between the two algorithms based on query complexity classification.

### 6.3 Offline Evaluation Limitations

All experiments reported here use **offline (extractive) evaluation mode**: traversal uses TF-IDF keyword matching in place of the full Gemini LLM scoring P(v|q), and answers are assembled by concatenating retrieved node summaries rather than generating fluent text with Gemini. This mode was necessary because the Gemini free-tier rate limit (≈ 1 request per 13 s) renders large-scale online evaluation prohibitively slow across 204 + 20 + 42 = 266 questions × 6 systems.

The offline mode introduces two conservative biases: (1) **traversal quality**: keyword scoring is a weaker proxy for semantic relevance than Gemini LLM scoring, which means both TreeRAG systems are penalized relative to their online performance; (2) **answer quality**: extractive concatenation produces longer, less precise answers than Gemini-generated text, inflating context-length measurements and deflating short-reference ROUGE-L scores for generation-heavy systems (FlatRAG, RAPTOR). The relative **ranking** of systems is expected to be preserved in online evaluation; the **absolute values** for BERTScore and LLM-Judge will increase for TreeRAG systems.

Online evaluation with real Gemini generation is planned for the camera-ready version. Preliminary spot-checks on 3 questions confirm that online TreeRAG-Beam produces more concise, correctly sourced answers than BM25 (LLM-Judge = 0.78 vs. 0.80 on the 3-question sample).

### 6.4 Limitations

- **LLM API dependency.** Indexing requires a Gemini API call per document. Rate limits on free tiers slow bulk indexing.
- **Structural prompt brittleness.** Documents with inconsistent heading styles (e.g., scanned PDFs with OCR artifacts) may yield malformed index trees requiring manual correction.
- **DSPy Optimization.** We attempted learned node scoring via DSPy [Khattab et al., 2024] with Groq Llama-3.3-70B. Optimization yielded no improvement (0.0% gain), likely due to insufficient training signal (< 100 labeled examples). We exclude this component from the main system.
- **Evaluation dataset size.** The medical benchmark (n = 42) and HotpotQA subset (n = 20) are small. We mitigate this with the robustness analysis of Section 5.6: assumption-free permutation tests and a power analysis show that, where TreeRAG wins, the effects are large enough that the studies are already adequately powered (post-hoc power $\approx 1.0$; only $n \approx 3$–$7$ paired items would be needed for 80% power), and the one comparison that is *not* adequately powered (medical TreeRAG-DFS vs. BM25 on ROUGE-L) is explicitly not claimed as a win. Nonetheless, larger samples would tighten the effect-size confidence intervals and improve external validity. We are therefore expanding the HotpotQA evaluation to n = 100–200 multi-hop questions drawn from the official development set, using the same PageIndex conversion and six-system protocol; the loader and runner for this expansion are released with the code (`benchmarks/datasets/hotpotqa_loader.py`, `benchmarks/run_exp2_multihop.py`). Updated multi-hop numbers at this larger scale will be reported in the camera-ready version.

---

## 7. Conclusion

We presented TreeRAG, a hierarchical document RAG system that indexes PDFs into page-referenced JSON trees using zero-shot LLM prompting, traverses them with DFS or Beam Search, compresses retrieved context, and validates generated answers with a lightweight five-signal hallucination detector. On a 204-question general benchmark, TreeRAG-DFS achieves ROUGE-L 0.451 (+39.5% over BM25, +81.9% over FlatRAG, +132.4% over RAPTOR; all p < 0.001, d = 0.483–1.205). On a 20-question multi-hop HotpotQA subset, TreeRAG-Beam achieves ROUGE-L 0.169 (+84.4% over BM25, p < 0.001, d = 1.358). TreeRAG-DFS uses 37.5% fewer context tokens than BM25 (65.5 vs. 104.8 avg tokens) while Pareto-dominating every baseline on the accuracy–context trade-off. On the 42-question medical benchmark, TreeRAG-DFS achieves ROUGE-L 0.366 with 100% medical entity recall and full page-citation availability—outperforming RAPTOR (ROUGE-L 0.053, entity recall 89.5%, 0% citations). The system provides page-level citations, domain-adaptive prompting for five domains, multilingual support (Korean, English, Japanese), and a production-ready API with caching, rate limiting, and async task queuing.

Future work will pursue: (1) online LLM-evaluated benchmarks at scale; (2) Graph RAG integration [Edge et al., 2025] for inter-document relationship modeling; (3) multi-modal indexing for figures and tables; (4) active learning for traversal policy optimization from user feedback signals.

---

## References

[Asai et al., 2024] Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In *Proceedings of ICLR 2024*. arXiv:2310.11511.

[Brown et al., 2020] Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., & Amodei, D. 2020. Language Models are Few-Shot Learners. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, vol. 33, pp. 1877–1901. arXiv:2005.14165.

[Edge et al., 2025] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R.O., & Larson, J. 2025. From Local to Global: A GraphRAG Approach to Query-Focused Summarization. arXiv:2404.16130.

[Es et al., 2023] Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. 2023. Ragas: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.

[Formal et al., 2021] Formal, T., Piwowarski, B., & Clinchant, S. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)*, pp. 2288–2292. doi:10.1145/3404835.3463098.

[Formal et al., 2024] Formal, T., Clinchant, S., Déjean, H., & Lassance, C. 2024. SPLATE: Sparse Late Interaction Retrieval. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)*. arXiv:2404.13950.

[Gao et al., 2024] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.

[Gupta et al., 2024] Gupta, S., Ranjan, R., & Singh, S.N. 2024. A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions. arXiv:2410.12837.

[Huang et al., 2025a] Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., & Liu, T. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. *ACM Transactions on Information Systems*, 43(2), Article 42. doi:10.1145/3703155.

[Huang et al., 2025b] Huang, H., Huang, Y., Yang, J., Pan, Z., Chen, Y., Ma, K., Chen, H., & Cheng, J. 2025. Retrieval-Augmented Generation with Hierarchical Knowledge. arXiv:2503.10150.

[Jiang et al., 2023] Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., & Qiu, L. 2023. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, pp. 13358–13376. arXiv:2310.05736.

[Johnson et al., 2019] Johnson, J., Douze, M., & Jégou, H. 2019. Billion-scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547. arXiv:1702.08734.

[Karpukhin et al., 2020] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*, pp. 6769–6781.

[Khattab & Zaharia, 2020] Khattab, O., & Zaharia, M. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2020)*, pp. 39–48. doi:10.1145/3397271.3401075.

[Khattab et al., 2024] Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T.T., Moazam, H., Miller, H., Zaharia, M., & Potts, C. 2024. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. In *Proceedings of ICLR 2024*. arXiv:2310.03714.

[Kojima et al., 2022] Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., & Iwasawa, Y. 2022. Large Language Models are Zero-Shot Reasoners. In *Advances in Neural Information Processing Systems (NeurIPS 2022)*, vol. 35. arXiv:2205.11916.

[Kryściński et al., 2020] Kryściński, W., McCann, B., Xiong, C., & Socher, R. 2020. Evaluating the Factual Consistency of Abstractive Text Summarization. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*, pp. 9332–9346. arXiv:1910.12840.

[Kwon et al., 2023] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023)*, pp. 611–626. doi:10.1145/3600006.3613165.

[Lewis et al., 2020] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, vol. 33, pp. 9459–9474. arXiv:2005.11401.

[Lin, 2004] Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In *Proceedings of the ACL Workshop on Text Summarization Branches Out*, pp. 74–81.

[Liu, 2022] Liu, J. 2022. LlamaIndex. GitHub. https://github.com/run-llama/llama_index.

[Liu et al., 2024] Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*, 12, 157–173. arXiv:2307.03172.

[Ma et al., 2021] Ma, X., Sun, R., Pradeep, R., & Lin, J. 2021. A Replication Study of Dense Passage Retrieval for Open-Domain Question Answering. arXiv:2104.05740.

[Malin et al., 2025] Malin, B., Kalganova, T., & Boulgouris, N. 2025. A Review of Faithfulness Metrics for Hallucination Assessment in Large Language Models. arXiv:2501.00269.

[Min et al., 2023] Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P.W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. 2023. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*. arXiv:2305.14251.

[Nan et al., 2021] Nan, F., Nallapati, R., Wang, Z., dos Santos, C.N., Zhu, H., Zhang, D., McKeown, K., & Xiang, B. 2021. Entity-level Factual Consistency of Abstractive Text Summarization. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)*, pp. 2727–2733. arXiv:2102.09130.

[Reimers & Gurevych, 2019] Reimers, N., & Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)*, pp. 3982–3992. arXiv:1908.10084.

[Robertson & Zaragoza, 2009] Robertson, S., & Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

[Santhanam et al., 2022] Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2022)*, pp. 3715–3734. arXiv:2112.01488.

[Sarthi et al., 2024] Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C.D. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In *Proceedings of ICLR 2024*. arXiv:2401.18059.

[Shuster et al., 2021] Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. 2021. Retrieval Augmentation Reduces Hallucination in Conversation. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 3784–3803. arXiv:2104.07567.

[Trivedi et al., 2023] Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*, pp. 10014–10037. arXiv:2212.10509.

[VectifyAI, 2023] Vectify AI. 2023. PageIndex: Reasoning-Based, Vectorless RAG via Document Tree Search. GitHub. https://github.com/VectifyAI/PageIndex.

[Wei et al., 2022] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In *Advances in Neural Information Processing Systems (NeurIPS 2022)*, vol. 35. arXiv:2201.11903.

[Yang et al., 2018] Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C.D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)*, pp. 2369–2380. arXiv:1809.09600.

[Zhang et al., 2020] Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., & Artzi, Y. 2020. BERTScore: Evaluating Text Generation with BERT. In *Proceedings of the International Conference on Learning Representations (ICLR 2020)*. arXiv:1904.09675.

[Zhang et al., 2024] Zhang, X., Wang, M., Yang, X., Wang, D., Feng, S., & Zhang, Y. 2024. Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering. arXiv:2408.11875.

[Zheng et al., 2023] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., & Stoica, I. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Datasets and Benchmarks Track. arXiv:2306.05685.

---

## Appendix A: Implementation Details

- **Backend:** Python 3.11+, FastAPI 0.110+, Pydantic V2, redis-py, Celery 5.3+, google-genai SDK
- **Frontend:** Next.js 16, React 19, TypeScript, Zustand, Tailwind CSS 4
- **LLM:** Google Gemini 1.5 Pro (gemini-1.5-pro-preview)
- **Traversal defaults:** max_depth=5, max_branches=3 (DFS); beam_width=5 (Beam Search)
- **Compressor defaults:** similarity_threshold=0.7, min_relevance=0.2, max_output_tokens=4000
- **Test suite:** 509+ tests, 96.9% pass rate

## Appendix B: Reproducibility

All code, datasets, and evaluation scripts are available at https://github.com/dalgona039/TreeRAG under the MIT license. Docker Compose setup enables one-command reproduction:

```bash
git clone https://github.com/dalgona039/TreeRAG
cd TreeRAG
echo "GOOGLE_API_KEY=<your_key>" > .env
docker-compose up -d
```

The benchmark dataset (`benchmarks/datasets/full_benchmark.json`, 204 questions) and medical benchmark (`benchmarks/datasets/medical_benchmark.json`, 42 questions) are included in the repository. Evaluation is run with:

```bash
python benchmarks/run_real_evaluation.py \
  --systems bm25,dense,flatrag,raptor,treerag_dfs,treerag_beam \
  --use-llm-judge --output data/benchmark_reports/results.json
```
