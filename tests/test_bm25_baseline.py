"""Unit tests for the BM25 retrieval baseline (PHASE B-1)."""
import pytest

from src.core.bm25_baseline import BM25Retriever, tokenize


@pytest.fixture
def sample_index():
    return {
        "id": "ROOT",
        "title": "초음파 영상",
        "summary": "초음파 진단의 개요",
        "page_ref": "1-10",
        "children": [
            {
                "id": "CH1",
                "title": "압전 효과",
                "summary": "압전 소자에 전압을 가해 진동을 만드는 원리",
                "page_ref": "3-5",
            },
            {
                "id": "CH2",
                "title": "Doppler Effect",
                "summary": "Frequency shift used to measure blood flow velocity",
                "page_ref": "6-8",
            },
            {
                "id": "CH3",
                "title": "감쇠 현상",
                "summary": "거리와 주파수 증가에 따른 에너지 손실",
                "page_ref": "9-10",
            },
        ],
    }


def test_tokenize_korean():
    toks = tokenize("압전 효과의 원리!")
    assert "압전" in toks and "효과의" in toks


def test_tokenize_english_lowercased():
    assert tokenize("Doppler Effect") == ["doppler", "effect"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_indexes_all_nodes(sample_index):
    r = BM25Retriever(sample_index)
    assert len(r) == 4  # ROOT + 3 children


def test_rejects_non_dict():
    with pytest.raises(TypeError):
        BM25Retriever(["not", "a", "dict"])


def test_retrieve_returns_required_keys(sample_index):
    r = BM25Retriever(sample_index)
    hits = r.retrieve("압전", top_k=3)
    assert hits
    for key in ("title", "summary", "page_ref", "score"):
        assert key in hits[0]


def test_retrieve_korean_relevance(sample_index):
    r = BM25Retriever(sample_index)
    hits = r.retrieve("압전 효과 원리", top_k=1)
    assert hits[0]["id"] == "CH1"


def test_retrieve_english_relevance(sample_index):
    r = BM25Retriever(sample_index)
    hits = r.retrieve("blood flow velocity", top_k=1)
    assert hits[0]["id"] == "CH2"


def test_top_k_limits_results(sample_index):
    r = BM25Retriever(sample_index)
    assert len(r.retrieve("초음파", top_k=2)) <= 2


def test_empty_query_returns_empty(sample_index):
    r = BM25Retriever(sample_index)
    assert r.retrieve("", top_k=5) == []


def test_scores_are_sorted_descending(sample_index):
    r = BM25Retriever(sample_index)
    hits = r.retrieve("주파수 에너지 손실", top_k=4)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_no_api_calls_pure_keyword(sample_index):
    # Construction and retrieval must not require any client/network.
    r = BM25Retriever(sample_index)
    assert isinstance(r.retrieve("초음파", top_k=2), list)
