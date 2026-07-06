# Benchmark Results — ollama / llama3.1:8b

**Dataset**: govreport_benchmark.json  |  **Questions**: 40  |  **Seed**: 42  |  **Date**: 20260706_050604

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.132 | 0.242 | 0.770 | 31.543 |
| Dense Retrieval | 0.125 | 0.233 | 0.702 | 31.463 |
| FlatRAG | 0.146 | 0.272 | 0.780 | 34.029 |
| RAPTOR | 0.066 | 0.107 | 0.612 | 9.927 |
| TreeRAG-DFS | 0.104 | 0.200 | 0.730 | 78.890 |
| TreeRAG-Beam | 0.154 | 0.307 | 0.818 | 168.945 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0014 | +0.022 | 0.41 | ✓ |
| Dense Retrieval | 0.0005 | +0.029 | 0.60 | ✓ |
| FlatRAG | 0.2992 | +0.008 | 0.16 | ✗ |
| RAPTOR | 0.0000 | +0.088 | 2.44 | ✓ |
| TreeRAG-DFS | 0.0000 | +0.050 | 0.93 | ✓ |
