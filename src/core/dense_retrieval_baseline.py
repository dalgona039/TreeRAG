"""
Dense retrieval baseline (PHASE B-2 of the KCI publication plan).

Embeds every node (title + summary) of a PageIndex tree and retrieves by
cosine similarity. Mirrors :class:`BM25Retriever`'s interface::

    retriever = DenseRetriever(index_dict)
    hits = retriever.retrieve("질문", top_k=10)

Embedding backend resolution order:

1. ``sentence-transformers`` with ``jhgan/ko-sroberta-multitask`` (Korean).
2. Fallback to ``intfloat/multilingual-e5-base``.
3. A deterministic, dependency-free hashing embedder (used automatically when
   neither model can be loaded, e.g. offline / no GPU). This keeps the rest of
   the pipeline runnable without network access; swap in a real model locally
   for publication-quality numbers.

A custom ``embedder`` callable ``(List[str]) -> np.ndarray`` may be injected
(unit tests use random/mock vectors so they run without a model or GPU).

Similarity search uses a FAISS ``IndexFlatIP`` when ``faiss`` is installed and
falls back to a NumPy inner-product scan otherwise. Embeddings are cached to
``data/indices/{doc_hash}_dense_index.pkl``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

KO_MODEL = "jhgan/ko-sroberta-multitask"
FALLBACK_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CACHE_DIR = os.path.join("data", "indices")
_HASH_DIM = 256

_WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+|[一-鿿]")


def _flatten(node: Dict[str, Any], acc: List[Dict[str, Any]]) -> None:
    if not isinstance(node, dict):
        return
    acc.append(node)
    for child in node.get("children", []) or []:
        _flatten(child, acc)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (network-free fallback)."""

    def __init__(self, dim: int = _HASH_DIM):
        self.dim = dim

    def __call__(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in (t.lower() for t in _WORD_RE.findall(text or "")):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        return vecs


def _build_default_embedder() -> Callable[[List[str]], np.ndarray]:
    """Try real sentence-transformers models, else the hashing fallback."""
    try:  # pragma: no cover - exercised only when the lib + model are present
        from sentence_transformers import SentenceTransformer

        for name in (KO_MODEL, FALLBACK_MODEL):
            try:
                model = SentenceTransformer(name)
                return lambda texts: np.asarray(
                    model.encode(list(texts), show_progress_bar=False),
                    dtype="float32",
                )
            except Exception:
                continue
    except Exception:
        pass
    return HashingEmbedder()


class DenseRetriever:
    def __init__(
        self,
        index: Dict[str, Any],
        embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ):
        if not isinstance(index, dict):
            raise TypeError("index must be a parsed PageIndex JSON dict")
        self.index = index
        self.embedder = embedder or _build_default_embedder()
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None

        self.nodes: List[Dict[str, Any]] = []
        _flatten(index, self.nodes)
        self._texts = [
            f"{n.get('title', '')} {n.get('summary', '')}".strip() for n in self.nodes
        ]

        self.embeddings = self._load_or_build_embeddings()
        self._faiss_index = self._build_faiss(self.embeddings)

    # ------------------------------------------------------------------ #
    def _doc_hash(self) -> str:
        payload = json.dumps(self.index, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]

    def _cache_path(self) -> Optional[str]:
        if not self.use_cache:
            return None
        return os.path.join(self.cache_dir, f"{self._doc_hash()}_dense_index.pkl")

    def _load_or_build_embeddings(self) -> np.ndarray:
        path = self._cache_path()
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("n") == len(self.nodes):
                    return np.asarray(cached["embeddings"], dtype="float32")
            except Exception:
                pass

        if not self.nodes:
            return np.zeros((0, _HASH_DIM), dtype="float32")

        raw = self.embedder(self._texts)
        embeddings = _normalize(raw)

        if path:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump({"n": len(self.nodes), "embeddings": embeddings}, f)
            except Exception:
                pass
        return embeddings

    def _build_faiss(self, embeddings: np.ndarray):
        if embeddings.shape[0] == 0:
            return None
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.ascontiguousarray(embeddings))
            return index
        except Exception:
            return None  # NumPy fallback handled in retrieve()

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.nodes or not query:
            return []

        q = _normalize(self.embedder([query]))
        k = min(max(top_k, 0), len(self.nodes))
        if k == 0:
            return []

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(np.ascontiguousarray(q), k)
            order = idx[0]
            sims = scores[0]
        else:
            sims_all = (self.embeddings @ q[0]).astype("float32")
            order = np.argsort(-sims_all)[:k]
            sims = sims_all[order]

        results: List[Dict[str, Any]] = []
        for rank, i in enumerate(order):
            if i < 0:
                continue
            node = self.nodes[int(i)]
            results.append(
                {
                    "id": node.get("id", ""),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "page_ref": node.get("page_ref", ""),
                    "score": float(sims[rank]),
                }
            )
        return results
