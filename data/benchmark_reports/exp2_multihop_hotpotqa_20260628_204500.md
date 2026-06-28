# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 5  |  **Seed**: 42  |  **Date**: 20260628_204500

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.255 | 0.255 | 0.787 | 3.559 |
| Dense Retrieval | 0.491 | 0.491 | 0.773 | 1.964 |
| FlatRAG | 0.426 | 0.426 | 0.853 | 2.815 |
| RAPTOR | 0.385 | 0.385 | 0.813 | 3.947 |
| TreeRAG-DFS | 0.155 | 0.155 | 0.827 | 23.078 |
| TreeRAG-Beam | 0.155 | 0.155 | 0.880 | 0.002 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.1642 | -0.100 | 0.55 | ✗ |
| Dense Retrieval | 0.1869 | -0.336 | 1.21 | ✗ |
| FlatRAG | 0.2602 | -0.271 | 0.94 | ✗ |
| RAPTOR | 0.2916 | -0.230 | 0.78 | ✗ |
| TreeRAG-DFS | 0.5000 | +0.000 | 0.00 | ✗ |
