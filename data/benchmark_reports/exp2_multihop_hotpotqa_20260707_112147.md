# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 5  |  **Seed**: 42  |  **Date**: 20260707_112147

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.235 | 0.235 | 0.693 | 10.586 |
| Dense Retrieval | 0.236 | 0.236 | 0.840 | 6.895 |
| FlatRAG | 0.046 | 0.046 | 0.693 | 8.626 |
| RAPTOR | 0.231 | 0.231 | 0.747 | 6.892 |
| TreeRAG-DFS | 0.032 | 0.032 | 0.640 | 63.235 |
| TreeRAG-Beam | 0.226 | 0.226 | 0.760 | 65.830 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.3875 | -0.009 | 0.02 | ✗ |
| Dense Retrieval | 0.4046 | -0.010 | 0.02 | ✗ |
| FlatRAG | 0.3298 | +0.180 | 0.58 | ✗ |
| RAPTOR | 0.4320 | -0.005 | 0.01 | ✗ |
| TreeRAG-DFS | 0.3182 | +0.193 | 0.63 | ✗ |
