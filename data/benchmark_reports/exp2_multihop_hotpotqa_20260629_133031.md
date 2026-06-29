# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 20  |  **Seed**: 42  |  **Date**: 20260629_133031

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.481 | 0.481 | 0.813 | 2.800 |
| Dense Retrieval | 0.498 | 0.503 | 0.807 | 2.667 |
| FlatRAG | 0.468 | 0.474 | 0.800 | 3.523 |
| RAPTOR | 0.597 | 0.602 | 0.740 | 2.290 |
| TreeRAG-DFS | 0.148 | 0.155 | 0.800 | 18.725 |
| TreeRAG-Beam | 0.148 | 0.155 | 0.833 | 0.002 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0112 | -0.334 | 1.36 | ✓ |
| Dense Retrieval | 0.0115 | -0.350 | 1.35 | ✓ |
| FlatRAG | 0.0117 | -0.320 | 1.29 | ✓ |
| RAPTOR | 0.0076 | -0.449 | 1.67 | ✓ |
| TreeRAG-DFS | 0.5000 | +0.000 | 0.00 | ✗ |
