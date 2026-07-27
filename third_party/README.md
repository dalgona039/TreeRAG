# third_party/raptor

Vendored copy of the official RAPTOR reference implementation
(https://github.com/parthsarthi03/raptor, MIT License, Parth Sarthi), commit
pulled 2026-07-27. Vendored rather than pip-installed because the upstream repo
ships no `setup.py`/`pyproject.toml` and is not published to PyPI under this
name (the `raptor` PyPI package is an unrelated tool). Unmodified from upstream
except for `LICENSE.txt` (copied alongside for attribution) and a one-line
guard in `cluster_utils.py::global_cluster_embeddings` that clamps UMAP's
`n_neighbors` to >=2 — upstream's unguarded `int(sqrt(n-1))` is 0 or 1 for
small leaf counts and hung indefinitely on that input during local testing.

See `src/core/raptor_baseline.py` for the local Ollama/sentence-transformers
adapter (`OllamaQAModel`, `OllamaSummarizationModel`) that lets this run
without an OpenAI API key.
