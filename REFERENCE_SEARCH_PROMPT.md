# TreeRAG 논문 레퍼런스 탐색 프롬프트

아래 프롬프트를 Semantic Scholar, Google Scholar, Perplexity, 또는 Claude/GPT에 붙여넣어 사용하세요.

---

## 사용 방법

1. **빠른 검색**: 아래 섹션 중 필요한 카테고리만 잘라서 검색
2. **전체 검색**: 전체 프롬프트를 AI에게 제공하여 한 번에 탐색
3. **검증**: 찾은 논문의 제목·저자·연도·venue를 반드시 Semantic Scholar (semanticscholar.org) 또는 ACL Anthology에서 직접 확인

---

## ──────────────────────────────────────────────
## PROMPT (복사해서 사용)
## ──────────────────────────────────────────────

```
You are a research assistant helping fill in references for an academic paper titled:

"TreeRAG: Hierarchical Document Intelligence via LLM-Driven Tree Indexing and Adaptive Traversal"

This paper proposes a RAG system that converts PDFs into hierarchical JSON trees using zero-shot LLM prompting,
traverses those trees with DFS or Beam Search, applies contextual compression, and detects hallucinations via
five lexical overlap signals. It is submitted to ACM SIGIR or CIKM 2026.

For EACH reference request below, provide:
  - Full title
  - All authors (surname, initials)
  - Year
  - Venue (conference/journal name + volume/issue if applicable)
  - arXiv ID or DOI if available
  - One-sentence summary of why it is relevant to TreeRAG

Be precise: if you are not certain of a detail, say so rather than guessing.

═══════════════════════════════════════════════════════════
SECTION A — FILL [CITE] PLACEHOLDERS (required for submission)
═══════════════════════════════════════════════════════════

A1. LLM hallucination survey
    Context in paper: "LLMs are prone to hallucination—generating plausible yet factually incorrect text [CITE]"
    Find: A well-cited survey or empirical paper on hallucination in large language models (2022–2024).
    Candidate to verify: "Survey of Hallucination in Natural Language Generation" (Ji et al., 2023, ACM Computing Surveys)

A2. Standard flat-chunk RAG pipeline
    Context: "The dominant RAG pipeline chunks source documents into fixed-length text windows, embeds each
    chunk with a dense encoder, and retrieves the top-K nearest neighbors [CITE]"
    Find: The canonical paper describing the flat-chunk embedding RAG pipeline.
    Candidates to verify:
      - Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (arXiv 2312.10997)
      - Izacard & Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain QA" (EACL 2021)

A3. HiRAG — graph-based hierarchical RAG
    Context: "HiRAG [CITE] ... explore graph-based hierarchies but require pre-built knowledge graphs"
    Find: A paper named or abbreviated "HiRAG" on hierarchical or graph-based retrieval-augmented generation.
    Search query: "HiRAG hierarchical graph retrieval augmented generation"

A4. HGTRAG — graph-based hierarchical RAG
    Context: "HGTRAG [CITE] ... graph-based hierarchies ... domain-specific schema engineering"
    Find: A paper named or abbreviated "HGTRAG".
    Search query: "HGTRAG heterogeneous graph retrieval augmented generation"
    Note: This may be a preprint or workshop paper — confirm it exists before citing.

═══════════════════════════════════════════════════════════
SECTION B — VERIFY & COMPLETE EXISTING PARTIAL CITATIONS
═══════════════════════════════════════════════════════════

B1. Lewis et al., 2020 — RAG paper
    Verify full citation:
    Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D.
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    NeurIPS 2020. Provide arXiv ID.

B2. Karpukhin et al., 2020 — Dense Passage Retrieval (DPR)
    Verify full citation:
    Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.
    "Dense Passage Retrieval for Open-Domain Question Answering"
    EMNLP 2020. Provide arXiv ID.

B3. Robertson & Zaragoza, 2009 — BM25
    Verify full citation:
    Robertson, S., & Zaragoza, H.
    "The Probabilistic Relevance Framework: BM25 and Beyond"
    Foundations and Trends in Information Retrieval, vol. 3, no. 4, 2009.

B4. Liu, 2022 — LlamaIndex
    Context: cited as a hierarchical tree-of-summarizations RAG system.
    Find: The correct technical report or paper for LlamaIndex (formerly GPT Index).
    Note: This is typically cited as a software/preprint, not a peer-reviewed paper.
    Candidate: Liu, J. (2022). LlamaIndex. GitHub. doi:10.5281/zenodo.1234 (check actual Zenodo/GitHub citation)

B5. Sarthi et al., 2024 — RAPTOR
    Verify full citation:
    Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C.D.
    "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
    ICLR 2024. Provide arXiv ID (arXiv:2401.18059).

B6. Trivedi et al., 2022 — IRCoT
    Context: mentioned in Section 2.3 as "IRCoT [Trivedi et al., 2022] interleaves retrieval and chain-of-thought"
    Verify full citation:
    Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A.
    "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
    ACL 2023 (note: may be 2022 arXiv, 2023 ACL). Provide arXiv ID.

B7. PageIndex / VectifyAI, 2023
    Context: cited as "PageIndex [VectifyAI, 2023] demonstrates vectorless, structure-first retrieval"
    Find: The blog post, GitHub repo, or technical report for the PageIndex framework by VectifyAI.
    This may be a blog/GitHub citation rather than a peer-reviewed paper — provide the URL.

B8. Johnson et al., 2019 — FAISS
    Verify full citation:
    Johnson, J., Douze, M., & Jégou, H.
    "Billion-Scale Similarity Search with GPUs"
    IEEE Transactions on Big Data, 2019 (published online 2017 as arXiv:1702.08734).

B9. Zhang et al., 2020 — BERTScore
    Verify full citation:
    Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., & Artzi, Y.
    "BERTScore: Evaluating Text Generation with BERT"
    ICLR 2020. arXiv:1904.09675.

B10. Yang et al., 2018 — HotpotQA
    Verify full citation:
    Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., & Manning, C.D.
    "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
    EMNLP 2018. arXiv:1809.09600.

B11. Lin, 2004 — ROUGE
    Verify full citation:
    Lin, C.-Y.
    "ROUGE: A Package for Automatic Evaluation of Summaries"
    ACL Workshop on Text Summarization Branches Out, 2004.

B12. Min et al., 2023 — FActScoring
    Verify full citation:
    Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P.W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H.
    "FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation"
    EMNLP 2023. arXiv:2305.14251.

B13. Es et al., 2023 — RAGAS
    Verify full citation:
    Es, S., James, J., Anke, L.E., & Schockaert, S.
    "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    arXiv:2309.15217, 2023.

B14. Shuster et al., 2021 — RAG reduces hallucination
    Verify full citation:
    Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J.
    "Retrieval Augmentation Reduces Hallucination in Conversation"
    EMNLP Findings 2021. arXiv:2104.07567.

B15. Khattab et al., 2023 — DSPy
    Verify full citation:
    Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., ... & Potts, C.
    "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
    arXiv:2310.03714, 2023 (ICLR 2024 version may exist — check).

═══════════════════════════════════════════════════════════
SECTION C — RECOMMENDED ADDITIONAL REFERENCES
(strengthen the paper; not currently cited but topically central)
═══════════════════════════════════════════════════════════

C1. RAG survey (2023–2024)
    Find: A comprehensive survey of RAG methods covering dense retrieval, sparse retrieval, and
    hybrid approaches, published 2023 or 2024.
    Candidate: Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey"
    arXiv:2312.10997, 2023.

C2. Context-length / token efficiency in RAG
    Find: A paper studying the effect of context length or context window on LLM generation quality in RAG.
    Keywords: "long context RAG", "context window retrieval", "lost in the middle"
    Candidate: Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (TACL 2024)

C3. Zero-shot prompting / in-context learning
    Context: TreeRAG uses zero-shot prompting for tree construction.
    Find: The seminal paper on zero-shot / few-shot prompting.
    Candidate: Brown et al., "Language Models are Few-Shot Learners" (NeurIPS 2020, GPT-3 paper)

C4. Sentence-BERT / bi-encoder for dense retrieval
    Context: Dense Retrieval baseline uses sentence-transformer embeddings.
    Find: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    EMNLP 2019. arXiv:1908.10084.

C5. Contextual compression in RAG
    Find: A paper or system paper describing context compression for RAG (reducing retrieved passage length).
    Keywords: "contextual compression retrieval", "RAG compression", "passage filtering"
    Note: LangChain's ContextualCompressionRetriever may only have blog/docs citations — check if a paper exists.

C6. Medical NLP / Clinical QA
    Context: TreeRAG has a medical domain benchmark.
    Find: A representative paper on RAG applied to clinical or biomedical documents.
    Candidates:
      - Singhal et al., "Large Language Models Encode Clinical Knowledge" (Nature 2023)
      - Zakka et al., "Almanac: Retrieval-Augmented Language Models for Clinical Medicine" (NEJM AI 2024)

C7. LLM-as-Judge evaluation methodology
    Context: TreeRAG uses Gemini as an automated judge (faithfulness, relevance, completeness).
    Find: A paper establishing or analyzing LLM-as-Judge for NLG evaluation.
    Candidate: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
    NeurIPS 2023. arXiv:2306.05685.

C8. Self-RAG or adaptive retrieval
    Find: A paper on adaptive or selective retrieval (deciding when to retrieve vs. answer from parametric memory).
    Candidate: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
    ICLR 2024. arXiv:2310.11511.

C9. Graph RAG
    Context: TreeRAG Discussion mentions Graph RAG integration as future work.
    Find: The Microsoft Graph RAG paper or a leading knowledge-graph-augmented RAG paper.
    Candidate: Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
    arXiv:2404.16130, 2024.

C10. Structured document understanding / PDF parsing
    Context: TreeRAG ingests PDFs and extracts hierarchical structure.
    Find: A paper on document layout analysis, PDF parsing, or structured document understanding.
    Candidates:
      - Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (ACM MM 2022)
      - arXiv papers on PDF structure extraction with LLMs (2023–2024)

═══════════════════════════════════════════════════════════
SECTION D — KOREAN/KCI ADDITIONAL REFERENCES
(if also targeting a Korean journal version)
═══════════════════════════════════════════════════════════

D1. Korean RAG / NLP paper
    Find: A Korean-language or Korea-authored paper on RAG, question answering, or document retrieval
    published in KIISE, KIPS, or KSBE journals (2022–2025).
    Keywords: "검색 증강 생성", "문서 질의응답", "정보 검색"

D2. KorQuAD
    Find: The KorQuAD 2.0 dataset paper for Korean machine reading comprehension.
    Candidate: Kim et al., "KorQuAD 2.0: Korean QA Dataset for Web Document Machine Comprehension"
    (verify venue and year)

D3. Korean biomedical NLP
    Context: TreeRAG was tested on Korean biomedical lecture materials (생체의공학개론).
    Find: A paper on Korean biomedical text processing or clinical NLP.
    Keywords: "한국어 의료", "biomedical Korean NLP"
```

---

## 검색 우선순위

| 우선도 | 섹션 | 이유 |
|--------|------|------|
| 🔴 필수 | A1, A3, A4 | `[CITE]` 플레이스홀더 — 없으면 제출 불가 |
| 🔴 필수 | B1–B6, B8–B15 | 기존 인용 검증 — venue/arXiv 오류 시 리뷰어 지적 |
| 🟡 권장 | C1, C2, C7, C9 | RAG survey, Lost in the Middle, LLM-judge, GraphRAG — 리뷰어가 기대하는 레퍼런스 |
| 🟡 권장 | C3, C4, C6, C8 | 방법론 근거 강화 (zero-shot, sentence-BERT, medical, Self-RAG) |
| 🟢 선택 | B7, C5, C10 | PageIndex URL, contextual compression, PDF parsing |
| 🟢 선택 | D1–D3 | KCI 버전 병행 투고 시 |

## 주의사항

- **A3(HiRAG), A4(HGTRAG)**: 검색해도 찾기 어려울 수 있음. 실제로 존재하지 않는 논문이라면 해당 `[CITE]`를 제거하거나 다른 graph-RAG 논문으로 대체
- **B7(PageIndex)**: 학술 논문이 아닌 GitHub/블로그일 가능성 높음 → `footnote`로 처리하거나 URL 인용
- **검색 도구**: Semantic Scholar API (api.semanticscholar.org), Google Scholar, ACL Anthology (aclanthology.org), arXiv

## 결과 반영 방법

찾은 레퍼런스는 `paper_draft.md`의 References 섹션 `[TBD]` 항목들을 아래 ACM 형식으로 교체:

```
[저자 성, 이니셜, 연도] 성, I., 성2, I2., ... 연도. 논문 제목. In *Venue* (Vol./No.), 페이지. DOI/arXiv.
```
