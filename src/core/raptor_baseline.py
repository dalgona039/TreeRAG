"""
RAPTOR baseline (PHASE 1-1 of the ACM upgrade plan).

RAPTOR (Sarthi et al., ICLR 2024) builds a *bottom-up* clustering tree by
recursively summarising clusters of chunks, then retrieves by traversing that
tree. It is the closest competing method to TreeRAG and the most important
baseline for the ACM submission.

This module wraps the real RAPTOR library — vendored at ``third_party/raptor``
from https://github.com/parthsarthi03/raptor (it isn't on PyPI under this
name; see ``third_party/README.md``) and driven by a local Ollama QA/
summarization adapter (``src/core/raptor_ollama_adapter.py``) instead of
RAPTOR's default OpenAI backend — and falls back to :class:`RaptorFallback`,
a deterministic, network-free approximation with no real clustering or LLM
summarization, only if the real library or its dependencies (torch,
sentence-transformers, umap-learn, faiss, scikit-learn) aren't importable, or
if the local Ollama server isn't reachable. Verified end-to-end (real GMM
clustering + genuine LLM-generated cluster summaries) as of 2026-07-27; see
``docs/paper_revision/B3_baseline_hyperparams.md`` for the prior state.
Results in ``data/benchmark_reports/`` predating that date used the fallback.

All retrieval results use the same node-dict schema as the other baselines:
``{"title", "summary", "page_ref", "score"}``.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

CHUNK_SIZE = 300
CLUSTER_SIZE = 3
SUMMARY_PREFIX_CHARS = 80

_PAGE_RE = re.compile(r"^---\s*PAGE\s+(\d+)\s*---\s*$")


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _split_pages(document_text: str) -> List[Dict[str, Any]]:
    """Split marker-annotated text into [{'page': int, 'text': str}, ...]."""
    pages: List[Dict[str, Any]] = []
    current_page = 1
    buffer: List[str] = []

    def flush():
        if buffer:
            pages.append({"page": current_page, "text": " ".join(buffer).strip()})

    for line in (document_text or "").splitlines():
        m = _PAGE_RE.match(line.strip())
        if m:
            flush()
            buffer = []
            current_page = int(m.group(1))
        else:
            if line.strip():
                buffer.append(line.strip())
    flush()
    if not pages and document_text:
        pages = [{"page": 1, "text": document_text.strip()}]
    return pages


def _chunk_text(document_text: str, size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
    """Split into ~``size``-char chunks, each tagged with its source page."""
    chunks: List[Dict[str, Any]] = []
    for seg in _split_pages(document_text):
        text, page = seg["text"], seg["page"]
        if not text:
            continue
        for i in range(0, len(text), size):
            piece = text[i : i + size].strip()
            if piece:
                chunks.append({"text": piece, "page": page})
    return chunks


def _char_trigrams(text: str) -> set:
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _overlap_score(query: str, node_text: str) -> float:
    """Deterministic character-trigram Jaccard overlap in [0, 1]."""
    q, n = _char_trigrams(query), _char_trigrams(node_text)
    if not q or not n:
        return 0.0
    inter = len(q & n)
    union = len(q | n)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #
class RaptorFallback:
    """Offline RAPTOR approximation (deterministic bottom-up cluster tree)."""

    def __init__(self, document_text: str, document_name: str):
        self.document_name = document_name
        self.chunks = _chunk_text(document_text)
        self.nodes: List[Dict[str, Any]] = self._build_tree()

    def _build_tree(self) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []

        # Leaf (chunk) nodes.
        chunk_nodes: List[Dict[str, Any]] = []
        for i, ch in enumerate(self.chunks):
            chunk_nodes.append(
                {
                    "id": "chunk_{0}".format(i),
                    "level": 0,
                    "title": "Chunk {0}".format(i),
                    "summary": ch["text"],
                    "page_ref": str(ch["page"]),
                    "score": 0.0,
                }
            )

        # Cluster nodes via deterministic round-robin grouping of 3.
        n_clusters = (len(chunk_nodes) + CLUSTER_SIZE - 1) // CLUSTER_SIZE if chunk_nodes else 0
        cluster_nodes: List[Dict[str, Any]] = []
        for c in range(n_clusters):
            members = [chunk_nodes[j] for j in range(c, len(chunk_nodes), n_clusters)] if n_clusters else []
            if not members:
                continue
            synth = " ".join(m["summary"][:SUMMARY_PREFIX_CHARS] for m in members).strip()
            pages = sorted({m["page_ref"] for m in members}, key=lambda p: int(p) if p.isdigit() else 0)
            page_ref = pages[0] if len(pages) == 1 else "{0}-{1}".format(pages[0], pages[-1])
            cluster_nodes.append(
                {
                    "id": "cluster_{0}".format(c),
                    "level": 1,
                    "title": "Cluster {0}".format(c),
                    "summary": synth,
                    "page_ref": page_ref,
                    "score": 0.0,
                }
            )

        nodes.extend(cluster_nodes)
        nodes.extend(chunk_nodes)
        return nodes

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        scored = []
        for node in self.nodes:
            s = _overlap_score(query, node["summary"])
            item = dict(node)
            item["score"] = float(s)
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [n for n in scored[: max(top_k, 0)] if n["score"] > 0] or scored[: max(top_k, 0)]


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #
class RaptorBaseline:
    """Wraps RAPTOR for comparison with TreeRAG; falls back to RaptorFallback."""

    def __init__(self, document_text: str, document_name: str):
        self.document_text = document_text
        self.document_name = document_name
        self._impl = None
        self._backend = "fallback"
        try:
            self._impl = self._try_real_raptor(document_text)
            if self._impl is not None:
                self._backend = "raptor"
        except Exception:
            self._impl = None
        if self._impl is None:
            self._impl = RaptorFallback(document_text, document_name)

    @staticmethod
    def _try_real_raptor(document_text: str):
        """Attempt to build a real RAPTOR tree; return an adapter or None.

        Uses the vendored RAPTOR implementation (``third_party/raptor``) with
        local Ollama QA/summarization models and the Dense baseline's embedder
        (``src/core/raptor_ollama_adapter.py``) instead of RAPTOR's default
        OpenAI backend, so no OpenAI API key is required.
        """
        try:
            from src.core.raptor_ollama_adapter import (
                ensure_raptor_importable,
                make_embedding_model,
                make_qa_model,
                make_summarization_model,
            )

            ensure_raptor_importable()
            from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
        except Exception:
            return None

        config = RetrievalAugmentationConfig(
            qa_model=make_qa_model(),
            summarization_model=make_summarization_model(),
            embedding_model=make_embedding_model(),
            tb_max_tokens=100,
            tb_num_layers=5,
            tr_top_k=5,
        )
        ra = RetrievalAugmentation(config=config)
        ra.add_documents(document_text)

        class _RealAdapter:
            def __init__(self, ra_inst):
                self.ra = ra_inst

            def retrieve(self, query, top_k=10):
                selected_nodes, _context = self.ra.retriever.retrieve_information_collapse_tree(
                    query, top_k=top_k, max_tokens=3500
                )
                out = []
                for node in selected_nodes:
                    layer = self.ra.retriever.tree_node_index_to_layer.get(node.index)
                    out.append(
                        {
                            "title": f"RAPTOR L{layer}-N{node.index}",
                            "summary": node.text,
                            "page_ref": "",
                            "score": 1.0,
                        }
                    )
                return out

        return _RealAdapter(ra)

    @property
    def backend(self) -> str:
        return self._backend

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return ranked node dicts: title, summary, page_ref, score."""
        try:
            nodes = self._impl.retrieve(query, top_k=top_k)
        except Exception:
            nodes = []
        out = []
        for n in nodes:
            out.append(
                {
                    "title": n.get("title", ""),
                    "summary": n.get("summary", ""),
                    "page_ref": n.get("page_ref", ""),
                    "score": float(n.get("score", 0.0)),
                }
            )
        return out

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Extractive answer + provenance. RAPTOR answers carry no page citations."""
        t0 = time.perf_counter()
        nodes = self.retrieve(query, top_k=top_k)
        context = " ".join(n["summary"] for n in nodes).strip()
        # Bottom-up summaries are not page-traceable -> no [doc, p.X] markers.
        answer = context[:600].strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "answer": answer,
            "source_nodes": nodes,
            "context_tokens": int(len(context) / 4),
            "latency_ms": latency_ms,
        }
