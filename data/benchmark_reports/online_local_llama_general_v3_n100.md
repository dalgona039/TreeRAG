# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260702_102213

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.372 | 0.395 | 0.781 | 12.449 |
| Dense Retrieval | 0.323 | 0.348 | 0.785 | 25.555 |
| FlatRAG | 0.405 | 0.440 | 0.786 | 30.659 |
| RAPTOR | 0.283 | 0.315 | 0.774 | 25.199 |
| TreeRAG-DFS | 0.308 | 0.360 | 0.792 | 158.702 |
| TreeRAG-Beam | 0.319 | 0.366 | 0.823 | 145.168 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0788 | -0.052 | 0.21 | ✗ |
| Dense Retrieval | 0.8708 | -0.004 | 0.02 | ✗ |
| FlatRAG | 0.0086 | -0.085 | 0.34 | ✓ |
| RAPTOR | 0.1717 | +0.036 | 0.15 | ✗ |
| TreeRAG-DFS | 0.5793 | +0.011 | 0.05 | ✗ |
