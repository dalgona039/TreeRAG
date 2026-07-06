# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 50  |  **Seed**: 42  |  **Date**: 20260706_200254

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.484 | 0.526 | 0.796 | 6.862 |
| Dense Retrieval | 0.406 | 0.449 | 0.788 | 7.216 |
| FlatRAG | 0.449 | 0.492 | 0.788 | 7.598 |
| RAPTOR | 0.358 | 0.387 | 0.800 | 7.465 |
| TreeRAG-DFS | 0.295 | 0.343 | 0.789 | 52.827 |
| TreeRAG-Beam | 0.317 | 0.367 | 0.799 | 26.635 |
| TreeRAG-Auto | 0.302 | 0.345 | 0.799 | 66.011 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0001 | -0.167 | 0.67 | ✓ |
| Dense Retrieval | 0.0132 | -0.089 | 0.34 | ✓ |
| FlatRAG | 0.0007 | -0.132 | 0.52 | ✓ |
| RAPTOR | 0.2432 | -0.041 | 0.17 | ✗ |
| TreeRAG-DFS | 0.0048 | +0.021 | 0.09 | ✓ |
| TreeRAG-Auto | 0.5748 | +0.014 | 0.06 | ✗ |
