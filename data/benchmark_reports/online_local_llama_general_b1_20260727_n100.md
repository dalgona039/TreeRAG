# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260727_233238

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.486 | 0.531 | 0.829 | 6.621 |
| Dense Retrieval | 0.000 | 0.000 | 0.804 | 1.612 |
| FlatRAG | 0.482 | 0.538 | 0.799 | 6.211 |
| RAPTOR | 0.392 | 0.435 | 0.790 | 60.097 |
| TreeRAG-DFS | 0.335 | 0.383 | 0.777 | 41.003 |
| TreeRAG-Beam | 0.351 | 0.416 | 0.806 | 78.738 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0000 | -0.135 | 0.56 | ✓ |
| Dense Retrieval | 0.0000 | +0.351 | 2.10 | ✓ |
| FlatRAG | 0.0000 | -0.131 | 0.53 | ✓ |
| RAPTOR | 0.0762 | -0.041 | 0.17 | ✗ |
| TreeRAG-DFS | 0.3782 | +0.016 | 0.07 | ✗ |
