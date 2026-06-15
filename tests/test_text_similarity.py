"""Unit tests for text-similarity metrics (PHASE C-1)."""
import pytest

from benchmarks.metrics.text_similarity import (
    batch_evaluate,
    bertscore_f1,
    exact_match,
    rouge_l_score,
)


# --- exact_match ----------------------------------------------------------
def test_exact_match_identical():
    assert exact_match("Hello World", "hello world") == 1.0


def test_exact_match_punctuation_normalized():
    assert exact_match("answer: 42!", "answer 42") == 1.0


def test_exact_match_different():
    assert exact_match("cat", "dog") == 0.0


def test_exact_match_both_empty():
    assert exact_match("", "") == 1.0


def test_exact_match_korean():
    assert exact_match("압전 효과.", "압전 효과") == 1.0


# --- rouge_l --------------------------------------------------------------
def test_rouge_l_perfect():
    assert rouge_l_score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)


def test_rouge_l_partial_between_0_and_1():
    s = rouge_l_score("the quick brown fox", "the slow brown dog")
    assert 0.0 < s < 1.0


def test_rouge_l_no_overlap():
    assert rouge_l_score("aaa bbb", "xxx yyy") == pytest.approx(0.0, abs=1e-6)


def test_rouge_l_empty_hypothesis():
    assert rouge_l_score("", "something") == 0.0


def test_rouge_l_both_empty():
    assert rouge_l_score("", "") == 1.0


def test_rouge_l_korean_partial():
    s = rouge_l_score("초음파 영상의 원리", "초음파 영상의 역사")
    assert 0.0 < s <= 1.0


# --- bertscore (offline proxy path) --------------------------------------
def test_bertscore_perfect():
    assert bertscore_f1("동일한 문장", "동일한 문장") == pytest.approx(1.0)


def test_bertscore_range():
    s = bertscore_f1("red apple fruit", "green apple fruit")
    assert 0.0 < s <= 1.0


def test_bertscore_empty():
    assert bertscore_f1("", "nonempty") == 0.0


# --- batch_evaluate -------------------------------------------------------
def test_batch_evaluate_shapes():
    hyps = ["a b c", "x y"]
    refs = ["a b c", "x z"]
    out = batch_evaluate(hyps, refs, ["rouge_l", "exact_match"])
    assert set(out) == {"rouge_l", "exact_match"}
    assert len(out["rouge_l"]) == 2
    assert out["exact_match"][0] == 1.0


def test_batch_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        batch_evaluate(["a"], ["a", "b"], ["rouge_l"])


def test_batch_evaluate_unknown_metric():
    with pytest.raises(ValueError):
        batch_evaluate(["a"], ["a"], ["nonexistent"])
