# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260707_022936

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.062 | 0.064 | 0.653 | 5.727 |
| Dense Retrieval | 0.064 | 0.066 | 0.628 | 4.277 |
| FlatRAG | 0.068 | 0.069 | 0.668 | 5.378 |
| RAPTOR | 0.054 | 0.054 | 0.641 | 5.227 |
| TreeRAG-DFS | 0.080 | 0.084 | 0.655 | 9.441 |
| TreeRAG-Beam | 0.104 | 0.106 | 0.616 | 6.935 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0894 | +0.042 | 0.22 | ✗ |
| Dense Retrieval | 0.0551 | +0.040 | 0.20 | ✗ |
| FlatRAG | 0.1441 | +0.035 | 0.17 | ✗ |
| RAPTOR | 0.0183 | +0.050 | 0.26 | ✓ |
| TreeRAG-DFS | 0.2473 | +0.024 | 0.11 | ✗ |
