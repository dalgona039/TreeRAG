"""Unit tests for the Gemini LLM-as-Judge evaluator (PHASE C-2).

Uses a fake client so no network/API access is required.
"""
import pytest

from benchmarks.metrics.llm_judge import GeminiJudge


class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc

    def generate_content(self, model, contents):
        if self._exc:
            raise self._exc
        return _Resp(self._text)


class _FakeClient:
    def __init__(self, text=None, exc=None):
        self.models = _FakeModels(text=text, exc=exc)


def _judge(text=None, exc=None):
    return GeminiJudge(client=_FakeClient(text=text, exc=exc), model="fake-model")


def test_perfect_scores_normalized():
    j = _judge('{"faithfulness":5,"relevance":5,"completeness":5,"reasoning":"great"}')
    out = j.score("q", "ctx", "ans", "exp")
    assert out["faithfulness"] == 1.0
    assert out["relevance"] == 1.0
    assert out["completeness"] == 1.0
    assert out["reasoning"] == "great"


def test_partial_scores_normalized():
    j = _judge('{"faithfulness":3,"relevance":4,"completeness":0,"reasoning":"ok"}')
    out = j.score("q", "ctx", "ans", "exp")
    assert out["faithfulness"] == pytest.approx(0.6)
    assert out["relevance"] == pytest.approx(0.8)
    assert out["completeness"] == 0.0


def test_scores_clamped_to_unit_range():
    j = _judge('{"faithfulness":9,"relevance":-2,"completeness":5,"reasoning":"x"}')
    out = j.score("q", "c", "a", "e")
    assert out["faithfulness"] == 1.0
    assert out["relevance"] == 0.0


def test_handles_code_fenced_json():
    j = _judge('```json\n{"faithfulness":5,"relevance":5,"completeness":5,"reasoning":"r"}\n```')
    out = j.score("q", "c", "a", "e")
    assert out["completeness"] == 1.0


def test_handles_json_embedded_in_prose():
    j = _judge('Sure! {"faithfulness":2,"relevance":2,"completeness":2,"reasoning":"r"} done')
    out = j.score("q", "c", "a", "e")
    assert out["faithfulness"] == pytest.approx(0.4)


def test_malformed_json_returns_none_scores():
    j = _judge("this is not json at all")
    out = j.score("q", "c", "a", "e")
    assert out["faithfulness"] is None
    assert out["relevance"] is None
    assert out["completeness"] is None


def test_api_exception_returns_none_scores():
    j = _judge(exc=RuntimeError("network down"))
    out = j.score("q", "c", "a", "e")
    assert all(out[a] is None for a in ("faithfulness", "relevance", "completeness"))
    assert "judge_error" in out["reasoning"]


def test_missing_axis_returns_none():
    j = _judge('{"faithfulness":5,"relevance":5,"reasoning":"no completeness"}')
    out = j.score("q", "c", "a", "e")
    assert out["faithfulness"] is None  # whole result invalidated


def test_non_numeric_score_returns_none():
    j = _judge('{"faithfulness":"high","relevance":5,"completeness":5,"reasoning":"r"}')
    out = j.score("q", "c", "a", "e")
    assert out["faithfulness"] is None


def test_reasoning_defaults_to_empty():
    j = _judge('{"faithfulness":5,"relevance":5,"completeness":5}')
    out = j.score("q", "c", "a", "e")
    assert out["reasoning"] == ""


def test_score_average_ok():
    j = _judge('{"faithfulness":5,"relevance":0,"completeness":5,"reasoning":"r"}')
    avg = j.score_average("q", "c", "a", "e")
    assert avg == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_score_average_none_on_failure():
    j = _judge("garbage")
    assert j.score_average("q", "c", "a", "e") is None
