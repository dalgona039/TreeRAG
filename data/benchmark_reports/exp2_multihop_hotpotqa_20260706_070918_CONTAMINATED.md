# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260706_070918

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.042 | 0.044 | 0.645 | 4.798 |
| Dense Retrieval | 0.054 | 0.057 | 0.673 | 4.313 |
| FlatRAG | 0.063 | 0.065 | 0.632 | 5.351 |
| RAPTOR | 0.050 | 0.050 | 0.625 | 4.949 |
| TreeRAG-DFS | 0.077 | 0.078 | 0.661 | 7.394 |
| TreeRAG-Beam | 0.077 | 0.078 | 0.679 | 0.001 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0237 | +0.036 | 0.27 | ✓ |
| Dense Retrieval | 0.2206 | +0.023 | 0.15 | ✗ |
| FlatRAG | 0.3166 | +0.015 | 0.09 | ✗ |
| RAPTOR | 0.0635 | +0.028 | 0.18 | ✗ |
| TreeRAG-DFS | 1.0000 | +0.000 | 0.00 | ✗ |
