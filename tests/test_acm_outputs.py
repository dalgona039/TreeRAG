"""Unit tests for PHASE 5 ACM table/summary generation. Offline."""
import pytest

from scripts.generate_paper_tables import (
    _citation_rate,
    table_main_combined,
    table_medical,
)
from scripts.generate_acm_outputs import contribution_summary

FULL = {
    "systems": ["bm25", "raptor", "treerag_beam"],
    "summary": {
        "bm25": {"rouge_l": 0.40, "bertscore": 0.45, "llm_judge": None, "latency": 0.003, "context_tokens": 100},
        "raptor": {"rouge_l": 0.10, "bertscore": 0.13, "llm_judge": None, "latency": 0.006, "context_tokens": 300},
        "treerag_beam": {"rouge_l": 0.55, "bertscore": 0.58, "llm_judge": None, "latency": 0.001, "context_tokens": 80},
    },
    "significance": {"raptor": {"p_value": 0.001, "significant": True}},
}
HOTPOT = {
    "systems": ["bm25", "raptor", "treerag_beam"],
    "summary": {
        "bm25": {"rouge_l": 0.09, "llm_judge": None},
        "raptor": {"rouge_l": 0.06, "llm_judge": None},
        "treerag_beam": {"rouge_l": 0.17, "llm_judge": None},
    },
}
MEDICAL = {
    "systems": ["bm25", "treerag_beam"],
    "summary": {
        "bm25": {"rouge_l": 0.40, "medical_entity_recall": 0.9},
        "treerag_beam": {"rouge_l": 0.30, "medical_entity_recall": 1.0},
    },
    "per_question": {
        "bm25": [{"answer": "no cite"}],
        "treerag_beam": [{"answer": "grounded [doc, p.5]"}],
    },
}


def test_citation_rate():
    assert _citation_rate([{"answer": "x [doc, p.3]"}, {"answer": "no"}]) == pytest.approx(0.5)


def test_table_main_combined_has_multirow_and_bolds_treerag():
    tex = table_main_combined(FULL, HOTPOT)
    assert r"\multirow" in tex and "HotpotQA" in tex
    assert r"\textbf{TreeRAG-Beam}" in tex
    assert r"\textbf{0.550}" in tex  # best full ROUGE-L bolded


def test_table_medical_columns():
    tex = table_medical(MEDICAL)
    assert "Entity Recall" in tex and "Citation Avail." in tex
    assert r"\textbf{1.000}" in tex  # best entity recall


def test_contribution_summary_mentions_all_benchmarks():
    text = contribution_summary(FULL, HOTPOT, MEDICAL)
    assert "full benchmark" in text.lower()
    assert "HotpotQA" in text
    assert "medical" in text.lower()
    assert "RAPTOR" in text
