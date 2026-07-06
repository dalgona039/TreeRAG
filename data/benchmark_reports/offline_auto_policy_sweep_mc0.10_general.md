# Benchmark Results — gemini / gemini-2.5-flash

**Dataset**: full_benchmark.json  |  **Questions**: 204  |  **Seed**: 0  |  **Date**: 20260706_163442

| System | ROUGE-L | BERTScore | Latency(s) |
|--------|---------|-----------|-----------|
| TreeRAG-DFS | 0.416 | 0.481 | 0.000 |
| TreeRAG-Beam | 0.393 | 0.458 | 0.000 |
| TreeRAG-Auto | 0.399 | 0.464 | 0.000 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| TreeRAG-DFS | 0.0000 | -0.023 | 0.10 | ✓ |
| TreeRAG-Auto | 0.0067 | -0.006 | 0.02 | ✓ |
