# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 40  |  **Seed**: 42  |  **Date**: 20260628_124015

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.305 | 0.318 | 0.778 | 6.644 |
| Dense Retrieval | 0.287 | 0.298 | 0.768 | 6.172 |
| FlatRAG | 0.386 | 0.403 | 0.775 | 7.443 |
| RAPTOR | 0.258 | 0.278 | 0.792 | 7.810 |
| TreeRAG-DFS | 0.252 | 0.291 | 0.785 | 51.555 |
| TreeRAG-Beam | 0.274 | 0.316 | 0.827 | 21.519 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.5394 | -0.031 | 0.12 | ✗ |
| Dense Retrieval | 0.8124 | -0.012 | 0.04 | ✗ |
| FlatRAG | 0.0525 | -0.111 | 0.38 | ✗ |
| RAPTOR | 0.7101 | +0.017 | 0.07 | ✗ |
| TreeRAG-DFS | 0.0409 | +0.022 | 0.10 | ✓ |
