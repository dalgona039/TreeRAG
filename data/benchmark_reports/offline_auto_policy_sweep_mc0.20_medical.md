# Benchmark Results — gemini / gemini-2.5-flash

**Dataset**: medical_benchmark.json  |  **Questions**: 42  |  **Seed**: 0  |  **Date**: 20260706_163458

| System | ROUGE-L | BERTScore | Med-Entity-Recall | Latency(s) |
|--------|---------|-----------|-----------------|-----------|
| TreeRAG-DFS | 0.366 | 0.398 | 1.000 | 0.000 |
| TreeRAG-Beam | 0.265 | 0.292 | 1.000 | 0.000 |
| TreeRAG-Auto | 0.280 | 0.306 | 1.000 | 0.000 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| TreeRAG-DFS | 0.0000 | -0.101 | 1.09 | ✓ |
| TreeRAG-Auto | 0.0140 | -0.015 | 0.17 | ✓ |
