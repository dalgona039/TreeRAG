# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260707_181812

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.275 | 0.276 | 0.755 | 6.912 |
| Dense Retrieval | 0.184 | 0.189 | 0.722 | 7.696 |
| FlatRAG | 0.213 | 0.214 | 0.753 | 8.245 |
| RAPTOR | 0.162 | 0.163 | 0.707 | 7.123 |
| TreeRAG-DFS | 0.141 | 0.141 | 0.742 | 53.585 |
| TreeRAG-Beam | 0.145 | 0.145 | 0.767 | 66.307 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0004 | -0.130 | 0.43 | ✓ |
| Dense Retrieval | 0.2101 | -0.039 | 0.15 | ✗ |
| FlatRAG | 0.0357 | -0.068 | 0.24 | ✓ |
| RAPTOR | 0.5988 | -0.017 | 0.07 | ✗ |
| TreeRAG-DFS | 0.8783 | +0.004 | 0.02 | ✗ |
