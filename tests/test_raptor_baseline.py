"""Unit tests for the RAPTOR baseline (PHASE 1). Uses mock text, no network."""
import pytest

from src.core.raptor_baseline import (
    RaptorBaseline,
    RaptorFallback,
    _chunk_text,
    _overlap_score,
    _split_pages,
)

DOC = (
    "--- PAGE 1 ---\n"
    "The heart pumps blood through the cardiovascular system. "
    "Cardiac output depends on heart rate and stroke volume.\n"
    "--- PAGE 2 ---\n"
    "Ultrasound imaging uses high frequency sound waves. "
    "The piezoelectric effect converts voltage into vibration.\n"
    "--- PAGE 3 ---\n"
    "Attenuation increases with distance and frequency in tissue."
)


def test_split_pages_tracks_page_numbers():
    pages = _split_pages(DOC)
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "cardiovascular" in pages[0]["text"]


def test_chunking_tags_pages():
    chunks = _chunk_text(DOC, size=40)
    assert chunks
    assert all("page" in c and "text" in c for c in chunks)
    assert {c["page"] for c in chunks} <= {1, 2, 3}


def test_overlap_score_range_and_self():
    assert _overlap_score("piezoelectric effect", "piezoelectric effect") == pytest.approx(1.0)
    assert 0.0 <= _overlap_score("heart blood", "cardiac output heart") <= 1.0
    assert _overlap_score("", "anything") == 0.0


def test_fallback_builds_cluster_and_chunk_nodes():
    fb = RaptorFallback(DOC, "doc")
    levels = {n["level"] for n in fb.nodes}
    assert 0 in levels and 1 in levels  # chunk + cluster levels


def test_fallback_is_deterministic():
    a = RaptorFallback(DOC, "doc").nodes
    b = RaptorFallback(DOC, "doc").nodes
    assert [n["id"] for n in a] == [n["id"] for n in b]
    assert [n["summary"] for n in a] == [n["summary"] for n in b]


def test_retrieve_returns_required_keys():
    r = RaptorBaseline(DOC, "doc")
    hits = r.retrieve("piezoelectric effect", top_k=3)
    assert hits
    for key in ("title", "summary", "page_ref", "score"):
        assert key in hits[0]


def test_retrieve_relevance():
    r = RaptorBaseline(DOC, "doc")
    hits = r.retrieve("piezoelectric effect vibration", top_k=1)
    assert "piezoelectric" in hits[0]["summary"].lower()


def test_retrieve_top_k_limit():
    r = RaptorBaseline(DOC, "doc")
    assert len(r.retrieve("heart", top_k=2)) <= 2


def test_backend_falls_back_offline():
    r = RaptorBaseline(DOC, "doc")
    assert r.backend == "fallback"  # real RAPTOR unavailable offline


def test_answer_schema_and_no_page_citation():
    r = RaptorBaseline(DOC, "doc")
    out = r.answer("What converts voltage into vibration?")
    for key in ("answer", "source_nodes", "context_tokens", "latency_ms"):
        assert key in out
    assert isinstance(out["context_tokens"], int)
    # RAPTOR answers are not page-traceable: no [doc, p.X] citation markers.
    import re

    assert not re.search(r"\[[^\]]*p\.\s*\d+", out["answer"])


def test_empty_document_does_not_crash():
    r = RaptorBaseline("", "empty")
    assert r.retrieve("anything", top_k=3) == []
    assert r.answer("anything")["context_tokens"] == 0
