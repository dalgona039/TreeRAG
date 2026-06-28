# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 40  |  **Seed**: 42  |  **Date**: 20260628_040353

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.294 | 0.331 | 0.697 | 0.004 |
| Dense Retrieval | 0.231 | 0.271 | 0.652 | 0.003 |
| FlatRAG | 0.261 | 0.291 | 0.802 | 0.004 |
| RAPTOR | 0.185 | 0.208 | 0.687 | 0.012 |
| TreeRAG-DFS | 0.118 | 0.130 | 0.695 | 48.795 |
| TreeRAG-Beam | 0.112 | 0.120 | 0.678 | 13.772 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0001 | -0.183 | 0.93 | ✓ |
| Dense Retrieval | 0.0050 | -0.119 | 0.63 | ✓ |
| FlatRAG | 0.0008 | -0.149 | 0.80 | ✓ |
| RAPTOR | 0.0525 | -0.073 | 0.43 | ✗ |
| TreeRAG-DFS | 0.1289 | -0.006 | 0.03 | ✗ |
