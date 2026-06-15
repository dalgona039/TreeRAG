"""
BM25 retrieval baseline (PHASE B-1 of the KCI publication plan).

A pure keyword-matching retriever over the *nodes* of a PageIndex JSON tree.
It mirrors the public interface expected by the evaluation runner:

    retriever = BM25Retriever(index_dict)
    hits = retriever.retrieve("질문", top_k=10)
    # -> [{"title": ..., "summary": ..., "page_ref": ..., "score": ...}, ...]

No external API is used. Tokenisation supports Korean (Hangul syllables),
Latin alphanumerics and CJK characters.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

# Hangul syllables, Latin/Greek/Cyrillic alphanumerics, and standalone CJK chars.
_WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+|[一-鿿]")


def tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer with Korean support.

    Splits on punctuation/whitespace, lowercases Latin text, and treats each
    CJK ideograph as its own token so Korean/Chinese queries match.
    """
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _flatten(node: Dict[str, Any], acc: List[Dict[str, Any]]) -> None:
    if not isinstance(node, dict):
        return
    acc.append(node)
    for child in node.get("children", []) or []:
        _flatten(child, acc)


class BM25Retriever:
    """BM25 over (title + summary) of every node in a PageIndex tree."""

    def __init__(self, index: Dict[str, Any]):
        if not isinstance(index, dict):
            raise TypeError("index must be a parsed PageIndex JSON dict")
        self.index = index

        self.nodes: List[Dict[str, Any]] = []
        _flatten(index, self.nodes)

        self._corpus_tokens: List[List[str]] = []
        for node in self.nodes:
            text = f"{node.get('title', '')} {node.get('summary', '')}".strip()
            tokens = tokenize(text)
            # BM25Okapi cannot handle empty documents gracefully for length
            # normalisation; insert a placeholder token so avgdl stays sane.
            self._corpus_tokens.append(tokens or ["<empty>"])

        self._bm25 = BM25Okapi(self._corpus_tokens) if self.nodes else None

    def __len__(self) -> int:
        return len(self.nodes)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return the ``top_k`` highest-scoring nodes for ``query``."""
        if self._bm25 is None or not query:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(self.nodes)), key=lambda i: scores[i], reverse=True
        )[: max(top_k, 0)]

        results: List[Dict[str, Any]] = []
        for i in ranked:
            node = self.nodes[i]
            results.append(
                {
                    "id": node.get("id", ""),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "page_ref": node.get("page_ref", ""),
                    "score": float(scores[i]),
                }
            )
        return results
