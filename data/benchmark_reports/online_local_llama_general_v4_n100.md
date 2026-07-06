# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260705_233902

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.473 | 0.530 | 0.826 | 20.897 |
| Dense Retrieval | 0.435 | 0.486 | 0.789 | 14.048 |
| FlatRAG | 0.479 | 0.528 | 0.798 | 16.255 |
| RAPTOR | 0.384 | 0.437 | 0.807 | 16.595 |
| TreeRAG-DFS | 0.340 | 0.389 | 0.785 | 93.871 |
| TreeRAG-Beam | 0.377 | 0.439 | 0.822 | 118.505 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0013 | -0.096 | 0.38 | ✓ |
| Dense Retrieval | 0.0322 | -0.058 | 0.23 | ✓ |
| FlatRAG | 0.0005 | -0.102 | 0.39 | ✓ |
| RAPTOR | 0.7699 | -0.007 | 0.03 | ✗ |
| TreeRAG-DFS | 0.0480 | +0.038 | 0.15 | ✓ |
