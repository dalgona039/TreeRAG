# Experiment 2: Multi-hop Quality — Category Breakdown

Source: `online_local_llama_general_v2.json`

## Factual

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.456 
| Dense Retrieval | 0.422 
| FlatRAG | 0.566 
| RAPTOR | 0.327 
| TreeRAG-DFS | 0.316 
| TreeRAG-Beam | 0.339 

## Multi Hop

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.117 
| Dense Retrieval | 0.162 
| FlatRAG | 0.233 
| RAPTOR | 0.198 
| TreeRAG-DFS | 0.193 
| TreeRAG-Beam | 0.204 

## Comparative

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.169 
| Dense Retrieval | 0.102 
| FlatRAG | 0.122 
| RAPTOR | 0.156 
| TreeRAG-DFS | 0.166 
| TreeRAG-Beam | 0.201 
