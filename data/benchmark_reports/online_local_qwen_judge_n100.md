# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260704_010718

| System | ROUGE-L | BERTScore | Latency(s) |
|--------|---------|-----------|-----------|
| BM25 | 0.352 | 0.382 | 6.870 |
| Dense Retrieval | 0.327 | 0.353 | 6.562 |
| FlatRAG | 0.411 | 0.451 | 12.149 |
| RAPTOR | 0.297 | 0.324 | 11.235 |
| TreeRAG-DFS | 0.311 | 0.357 | 48.881 |
| TreeRAG-Beam | 0.293 | 0.340 | 79.228 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0400 | -0.059 | 0.23 | ✓ |
| Dense Retrieval | 0.1781 | -0.034 | 0.13 | ✗ |
| FlatRAG | 0.0001 | -0.118 | 0.44 | ✓ |
| RAPTOR | 0.8699 | -0.004 | 0.02 | ✗ |
| TreeRAG-DFS | 0.4320 | -0.018 | 0.08 | ✗ |
