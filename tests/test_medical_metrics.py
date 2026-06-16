"""Unit tests for medical-domain additions (PHASE 3)."""
import pytest

from benchmarks.metrics.text_similarity import (
    MEDICAL_TERMS,
    batch_evaluate,
    medical_entity_recall,
)
from benchmarks.datasets.auto_qa_generator import (
    generate_offline_medical_questions,
)


def test_entity_recall_full_match():
    ref = "The cardiac output depends on heart rate."
    hyp = "Cardiac output is set by heart rate and stroke volume."
    assert medical_entity_recall(hyp, ref) == pytest.approx(1.0)


def test_entity_recall_partial():
    ref = "ultrasound uses the piezoelectric effect and impedance matching"
    hyp = "ultrasound imaging"
    score = medical_entity_recall(hyp, ref)
    assert 0.0 < score < 1.0


def test_entity_recall_no_terms_in_reference_is_one():
    assert medical_entity_recall("anything", "the quick brown fox") == 1.0


def test_entity_recall_korean_terms():
    # reference has medical terms 초음파 + 영상; hyp matches only 초음파 -> 0.5
    assert medical_entity_recall("초음파 진단", "초음파 영상의 원리") == pytest.approx(0.5)
    # hyp covering both terms -> full recall
    assert medical_entity_recall("초음파 영상 분석", "초음파 영상의 원리") == pytest.approx(1.0)


def test_entity_recall_registered_in_batch():
    out = batch_evaluate(["cardiac"], ["cardiac output"], ["medical_entity_recall"])
    assert out["medical_entity_recall"][0] == pytest.approx(1.0)


def test_medical_terms_nonempty():
    assert len(MEDICAL_TERMS) >= 50


def test_offline_medical_questions_categories():
    tree = {
        "id": "ROOT", "title": "doc", "summary": "s", "page_ref": "1",
        "children": [
            {"id": "A", "title": "초음파 원리", "summary": "초음파의 기본 원리 설명입니다.", "page_ref": "2"},
            {"id": "B", "title": "압전 효과", "summary": "압전 소자의 전압 변환 원리입니다.", "page_ref": "3"},
            {"id": "C", "title": "감쇠", "summary": "거리와 주파수에 따른 에너지 손실입니다.", "page_ref": "4"},
            {"id": "D", "title": "임피던스", "summary": "음향 임피던스와 반사의 관계입니다.", "page_ref": "5"},
        ],
    }
    qs = generate_offline_medical_questions(tree, n=14)
    cats = {q["category"] for q in qs}
    assert {"clinical_fact", "procedure", "comparison", "safety"} <= cats
    assert all("clinical_relevance" in q for q in qs)
