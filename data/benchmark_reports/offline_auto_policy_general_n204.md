# Benchmark Results — gemini / gemini-2.5-flash

**Dataset**: full_benchmark.json  |  **Questions**: 204  |  **Seed**: 0  |  **Date**: 20260706_163417

| System | ROUGE-L | BERTScore | Latency(s) |
|--------|---------|-----------|-----------|
| BM25 | 0.298 | 0.344 | 0.001 |
| Dense Retrieval | 0.267 | 0.310 | 0.000 |
| FlatRAG | 0.248 | 0.294 | 0.000 |
| RAPTOR | 0.194 | 0.219 | 0.002 |
| TreeRAG-DFS | 0.416 | 0.481 | 0.000 |
| TreeRAG-Beam | 0.393 | 0.458 | 0.000 |
| TreeRAG-Auto | 0.395 | 0.460 | 0.000 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0000 | +0.095 | 0.47 | ✓ |
| Dense Retrieval | 0.0000 | +0.126 | 0.65 | ✓ |
| FlatRAG | 0.0000 | +0.145 | 0.77 | ✓ |
| RAPTOR | 0.0000 | +0.199 | 1.10 | ✓ |
| TreeRAG-DFS | 0.0000 | -0.023 | 0.10 | ✓ |
| TreeRAG-Auto | 0.0948 | -0.002 | 0.01 | ✗ |
