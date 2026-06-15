"""Unit tests for the dense retrieval baseline (PHASE B-2).

Uses a deterministic mock embedder so tests run without a model or GPU.
"""
import numpy as np
import pytest

from src.core.dense_retrieval_baseline import DenseRetriever, HashingEmbedder


@pytest.fixture
def sample_index():
    return {
        "id": "ROOT",
        "title": "Document",
        "summary": "root summary",
        "page_ref": "1",
        "children": [
            {"id": "A", "title": "apple", "summary": "a red fruit", "page_ref": "2"},
            {"id": "B", "title": "banana", "summary": "a yellow fruit", "page_ref": "3"},
            {"id": "C", "title": "car", "summary": "a motor vehicle", "page_ref": "4"},
        ],
    }


@pytest.fixture
def hashing():
    # No-cache hashing embedder: deterministic and network-free.
    return HashingEmbedder()


def _retriever(index, embedder):
    return DenseRetriever(index, embedder=embedder, use_cache=False)


def test_builds_over_all_nodes(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    assert len(r) == 4


def test_rejects_non_dict(hashing):
    with pytest.raises(TypeError):
        DenseRetriever([1, 2, 3], embedder=hashing, use_cache=False)


def test_retrieve_keys(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    hits = r.retrieve("fruit", top_k=2)
    assert hits
    for key in ("title", "summary", "page_ref", "score"):
        assert key in hits[0]


def test_hashing_embedder_deterministic():
    e = HashingEmbedder()
    v1 = e(["압전 효과"])
    v2 = e(["압전 효과"])
    assert np.allclose(v1, v2)


def test_embeddings_are_normalized(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    norms = np.linalg.norm(r.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mock_embedder_controls_ranking(sample_index):
    # Mock embedder: query aligns exactly with node "B".
    dim = 4
    vectors = {
        "Document root summary": [1, 0, 0, 0],
        "apple a red fruit": [0, 1, 0, 0],
        "banana a yellow fruit": [0, 0, 1, 0],
        "car a motor vehicle": [0, 0, 0, 1],
        "find banana": [0, 0, 1, 0],
    }

    def embedder(texts):
        return np.array([vectors[t] for t in texts], dtype="float32")

    r = DenseRetriever(sample_index, embedder=embedder, use_cache=False)
    hits = r.retrieve("find banana", top_k=1)
    assert hits[0]["id"] == "B"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_top_k_limit(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    assert len(r.retrieve("fruit", top_k=2)) == 2


def test_top_k_capped_to_node_count(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    assert len(r.retrieve("fruit", top_k=100)) == 4


def test_empty_query_returns_empty(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    assert r.retrieve("", top_k=3) == []


def test_scores_sorted_descending(sample_index, hashing):
    r = _retriever(sample_index, hashing)
    hits = r.retrieve("a fruit vehicle", top_k=4)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_cache_roundtrip(tmp_path, sample_index):
    rng = np.random.default_rng(0)

    def embedder(texts):
        return rng.standard_normal((len(texts), 8)).astype("float32")

    r1 = DenseRetriever(sample_index, embedder=embedder, cache_dir=str(tmp_path))
    files = list(tmp_path.glob("*_dense_index.pkl"))
    assert files, "embedding cache file should be written"
    # Second build with a different embedder should load from cache (same shape).
    r2 = DenseRetriever(sample_index, embedder=embedder, cache_dir=str(tmp_path))
    assert np.allclose(r1.embeddings, r2.embeddings)
