"""Unit tests for the HotpotQA loader/converter (PHASE 2). Offline, no network."""
import pytest

from benchmarks.datasets.hotpotqa_loader import (
    _build_flat_tree,
    _normalize_item,
    convert_to_benchmark_format,
    load_hotpotqa_subset,
)

NATIVE = {
    "_id": "hp_x",
    "type": "bridge",
    "question": "What effect powers the transducer?",
    "answer": "The piezoelectric effect",
    "supporting_facts": [["Transducer", 0]],
    "context": [
        ["Transducer", ["It uses the piezoelectric effect.", "It emits waves."]],
        ["Other", ["Unrelated sentence."]],
    ],
}


def test_load_subset_falls_back_to_sample():
    items = load_hotpotqa_subset(n=10, seed=1)
    assert 1 <= len(items) <= 10
    assert all(it["type"] in ("comparison", "bridge") for it in items)


def test_subset_is_deterministic_with_seed():
    a = [i["question_id"] for i in load_hotpotqa_subset(n=10, seed=7)]
    b = [i["question_id"] for i in load_hotpotqa_subset(n=10, seed=7)]
    assert a == b


def test_normalize_item_schema():
    norm = _normalize_item(NATIVE)
    assert norm["supporting_facts"][0] == {"title": "Transducer", "sent_id": 0}
    assert norm["context"][0]["title"] == "Transducer"


def test_build_flat_tree_structure():
    tree = _build_flat_tree(_normalize_item(NATIVE), "doc")
    assert tree["id"] == "ROOT"
    assert len(tree["children"]) == 2
    assert tree["children"][0]["title"] == "Transducer"


def test_convert_sets_expected_sections_from_supporting_facts(tmp_path, monkeypatch):
    import benchmarks.datasets.hotpotqa_loader as mod

    monkeypatch.setattr(mod, "INDEX_DIR", tmp_path)
    ds = convert_to_benchmark_format([_normalize_item(NATIVE)], write_indices=True)
    q = ds["questions"][0]
    assert q["category"] == "multi_hop"
    assert q["expected_answer_hint"] == "The piezoelectric effect"
    assert "SEC0" in q["expected_sections"]  # the supporting "Transducer" section
    assert (tmp_path / q["document_id"]).exists()


def test_convert_total_count():
    items = load_hotpotqa_subset(n=20, seed=42)
    ds = convert_to_benchmark_format(items, write_indices=False)
    assert ds["total_questions"] == len(items)
    assert ds["backend"] == "hotpotqa"
