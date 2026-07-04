# Benchmark Results — ollama / llama3.1:8b

**Dataset**: govreport_benchmark.json  |  **Questions**: 40  |  **Seed**: 42  |  **Date**: 20260704_064358

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.038 | 0.061 | 0.825 | 42.062 |
| Dense Retrieval | 0.026 | 0.037 | 0.815 | 45.105 |
| FlatRAG | 0.146 | 0.282 | 0.757 | 34.575 |
| RAPTOR | 0.021 | 0.034 | 0.523 | 14.070 |
| TreeRAG-DFS | 0.119 | 0.228 | 0.735 | 81.047 |
| TreeRAG-Beam | 0.164 | 0.333 | 0.825 | 160.218 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0000 | +0.126 | 2.22 | ✓ |
| Dense Retrieval | 0.0000 | +0.138 | 2.83 | ✓ |
| FlatRAG | 0.1300 | +0.018 | 0.30 | ✗ |
| RAPTOR | 0.0000 | +0.143 | 2.97 | ✓ |
| TreeRAG-DFS | 0.0024 | +0.045 | 0.68 | ✓ |
