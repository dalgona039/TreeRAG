# Benchmark Results — ollama / llama3.1:8b

**Dataset**: medical_benchmark.json  |  **Questions**: 42  |  **Seed**: 0  |  **Date**: 20260729_193238

| System | ROUGE-L | BERTScore | LLM-Judge | Med-Entity-Recall | Latency(s) |
|--------|---------|-----------|-----------|-----------------|-----------|
| BM25 | 0.257 | 0.281 | 0.792 | 0.895 | 12.347 |
| Dense Retrieval | 0.257 | 0.281 | 0.814 | 0.954 | 13.416 |
| FlatRAG | 0.337 | 0.374 | 0.771 | 0.974 | 10.707 |
| RAPTOR | 0.073 | 0.091 | 0.727 | 0.825 | 53.575 |
| TreeRAG-DFS | 0.067 | 0.074 | 0.621 | 0.720 | 58.968 |
| TreeRAG-Beam | 0.117 | 0.138 | 0.795 | 0.893 | 143.436 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| BM25 | 0.0000 | -0.139 | 0.93 | ✓ |
| Dense Retrieval | 0.0001 | -0.140 | 0.80 | ✓ |
| FlatRAG | 0.0000 | -0.220 | 1.06 | ✓ |
| RAPTOR | 0.0073 | +0.044 | 0.50 | ✓ |
| TreeRAG-DFS | 0.0171 | +0.050 | 0.49 | ✓ |
