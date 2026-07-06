# Benchmark Results — ollama / llama3.1:8b

**Dataset**: full_benchmark.json  |  **Questions**: 50  |  **Seed**: 42  |  **Date**: 20260706_234655

| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) |
|--------|---------|-----------|-----------|-----------|
| TreeRAG-DFS | 0.321 | 0.367 | 0.812 | 53.364 |
| TreeRAG-Beam | 0.373 | 0.425 | 0.781 | 96.078 |
| TreeRAG-Auto | 0.293 | 0.355 | 0.779 | 82.570 |

## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)

| vs System | p-value | Δ mean | Cohen's d | Sig? |
|-----------|---------|--------|-----------|------|
| TreeRAG-DFS | 0.0524 | +0.052 | 0.21 | ✗ |
| TreeRAG-Auto | 0.0010 | +0.080 | 0.33 | ✓ |
