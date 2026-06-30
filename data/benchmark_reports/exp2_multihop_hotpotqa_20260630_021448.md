# Benchmark Results — ollama / llama3.1:8b

**Dataset**: hotpotqa  |  **Questions**: 100  |  **Seed**: 42  |  **Date**: 20260630_021448

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| BM25 | 0.122 | 0.123 | 0.574 | 4.391 |
| Dense Retrieval | 0.102 | 0.104 | 0.598 | 3.686 |
| FlatRAG | 0.098 | 0.098 | 0.614 | 5.546 |
| RAPTOR | 0.086 | 0.087 | 0.585 | 4.423 |
| TreeRAG-DFS | 0.065 | 0.067 | 0.649 | 8.310 |
| TreeRAG-Beam | 0.065 | 0.067 | 0.621 | 0.002 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0086 | -0.057 | 0.26 | ✓ |
| Dense Retrieval | 0.1094 | -0.037 | 0.18 | ✗ |
| FlatRAG | 0.0934 | -0.032 | 0.16 | ✗ |
| RAPTOR | 0.2549 | -0.021 | 0.11 | ✗ |
| TreeRAG-DFS | 1.0000 | +0.000 | 0.00 | ✗ |
