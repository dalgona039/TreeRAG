"""Local (no-OpenAI-key) model adapters for the vendored RAPTOR library.

The official RAPTOR implementation (``third_party/raptor``, vendored from
https://github.com/parthsarthi03/raptor) hard-codes OpenAI as its QA and
summarization backend and only offers a non-OpenAI option for embeddings
(``SBertEmbeddingModel``). This module supplies QA/summarization adapters that
call the same local Ollama backend already used elsewhere in this repo
(``src/core/ollama_client.py``), and a thin embedding wrapper that reuses the
Dense baseline's embedder-resolution logic so RAPTOR and Dense share the same
embedding backend for a fair comparison.

Import is deferred/lazy everywhere here so that importing this module never
requires Ollama to be running or `third_party/raptor` to be import-able;
failures surface only when a method is actually called.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

_THIRD_PARTY_DIR = str(Path(__file__).resolve().parents[2] / "third_party")


def ensure_raptor_importable() -> None:
    """Add the vendored ``third_party/raptor`` package to ``sys.path`` once."""
    import os

    # faiss (RAPTOR's vector index) and torch (sentence-transformers) each
    # bundle their own OpenMP runtime; loading both in one process on macOS
    # segfaults (EXC_BAD_ACCESS) without this well-known workaround. Set
    # before raptor/faiss are imported anywhere in the process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if _THIRD_PARTY_DIR not in sys.path:
        sys.path.insert(0, _THIRD_PARTY_DIR)


_shared_embed_fn = None
_shared_embed_lock = threading.Lock()


def _get_shared_embed_fn():
    """Lazily build (once per process) the same embedder DenseRetriever uses.

    RAPTOR embeds leaf nodes concurrently via ThreadPoolExecutor
    (tree_builder.py's default ``use_multithreading=True``), so without this
    lock every worker thread would race past the ``is None`` check and each
    load its own full sentence-transformers model at once — this previously
    caused RAPTOR tree-building to hang/OOM on a 20-leaf test document.
    """
    global _shared_embed_fn
    if _shared_embed_fn is None:
        with _shared_embed_lock:
            if _shared_embed_fn is None:
                from src.core.dense_retrieval_baseline import _build_default_embedder

                _shared_embed_fn = _build_default_embedder()
    return _shared_embed_fn


class _LazyOllama:
    """Shared lazy-init Ollama client, one instance per process."""

    _client: Optional[Any] = None

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    def _client_ref(self):
        if _LazyOllama._client is None:
            from src.core.ollama_client import OllamaClient

            _LazyOllama._client = OllamaClient(base_url=self._base_url, model=self._model)
        return _LazyOllama._client

    def generate(self, prompt: str, stage_name: str = "raptor_index") -> str:
        from src.utils import llm_accounting

        with llm_accounting.stage(stage_name):
            resp = self._client_ref().models.generate_content(model=None, contents=prompt, config=None)
        return (resp.text or "").strip()


# ``RetrievalAugmentationConfig`` does ``isinstance(x, BaseQAModel)`` etc., so
# these adapters must actually subclass RAPTOR's ABCs. Those ABCs only become
# importable after ``ensure_raptor_importable()`` has run, so the classes are
# built lazily inside factory functions rather than at module import time.


def make_embedding_model():
    """RAPTOR ``BaseEmbeddingModel`` backed by the Dense baseline's embedder."""
    ensure_raptor_importable()
    from raptor import BaseEmbeddingModel

    class LocalSBertEmbeddingModel(BaseEmbeddingModel):
        def create_embedding(self, text: str) -> List[float]:
            import numpy as np

            vec = _get_shared_embed_fn()([text])
            return np.asarray(vec)[0].tolist()

    return LocalSBertEmbeddingModel()


def make_qa_model(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
    """RAPTOR ``BaseQAModel`` backed by a local Ollama model instead of OpenAI."""
    ensure_raptor_importable()
    from raptor import BaseQAModel

    class OllamaQAModel(BaseQAModel):
        def __init__(self):
            self._ollama = _LazyOllama(model=model, base_url=base_url)

        def answer_question(self, context: str, question: str) -> str:
            prompt = (
                "Answer the question using ONLY the context below. "
                "Do NOT speculate beyond the context. Be concise.\n\n"
                f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"
            )
            try:
                return self._ollama.generate(prompt, stage_name="raptor_index_qa")
            except Exception:
                return ""

    return OllamaQAModel()


def make_summarization_model(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
    """RAPTOR ``BaseSummarizationModel`` backed by a local Ollama model."""
    ensure_raptor_importable()
    from raptor import BaseSummarizationModel

    class OllamaSummarizationModel(BaseSummarizationModel):
        def __init__(self):
            self._ollama = _LazyOllama(model=model, base_url=base_url)

        def summarize(self, context: str, max_tokens: int = 150) -> str:
            prompt = (
                "Write a concise summary of the following text, including as many "
                f"key details as possible (roughly {max_tokens} tokens or fewer):\n\n"
                f"{context}\n\nSummary:"
            )
            try:
                return self._ollama.generate(prompt, stage_name="raptor_index_summarization")
            except Exception:
                return context[: max_tokens * 4]

    return OllamaSummarizationModel()
