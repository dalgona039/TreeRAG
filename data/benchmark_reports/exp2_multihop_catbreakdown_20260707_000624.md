# Experiment 2: Multi-hop Quality — Category Breakdown

Source: `online_local_llama_general_v4_n100.json`

## Factual

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.569 
| Dense Retrieval | 0.515 
| FlatRAG | 0.615 
| RAPTOR | 0.437 
| TreeRAG-DFS | 0.408 
| TreeRAG-Beam | 0.454 

## Multi Hop

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.373 
| Dense Retrieval | 0.362 
| FlatRAG | 0.357 
| RAPTOR | 0.332 
| TreeRAG-DFS | 0.289 
| TreeRAG-Beam | 0.300 

## Comparative

| System | ROUGE-L | LLM-Judge | N |
|--------|---------|-----------|---|
| BM25 | 0.390 
| Dense Retrieval | 0.349 
| FlatRAG | 0.336 
| RAPTOR | 0.336 
| TreeRAG-DFS | 0.250 
| TreeRAG-Beam | 0.308 
