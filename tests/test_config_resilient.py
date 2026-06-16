"""Tests for the resilient Gemini client wrapper (429 retry + throttle)."""
import pytest

import src.config as cfg


class _Resp:
    def __init__(self, text):
        self.text = text


class _FlakyInner:
    """Fails with a 429-style error `fail_times` times, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def generate_content(self, *a, **k):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, retryDelay: '2s'")
        return _Resp("OK")


def test_is_rate_limit_error():
    assert cfg._is_rate_limit_error("429 RESOURCE_EXHAUSTED")
    assert cfg._is_rate_limit_error("Quota exceeded")
    assert not cfg._is_rate_limit_error("400 invalid argument")


def test_retry_delay_parsed():
    assert cfg._retry_delay_seconds("please retryDelay: '30s'") == 30.0
    assert cfg._retry_delay_seconds("no hint here") is None


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(cfg, "_GEMINI_MAX_RETRIES", 5)
    monkeypatch.setattr(cfg, "_GEMINI_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(cfg._time, "sleep", lambda *_: None)  # no real waiting
    models = cfg._ResilientModels(_FlakyInner(fail_times=2))
    resp = models.generate_content(model="m", contents="hi")
    assert resp.text == "OK"


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(cfg, "_GEMINI_MAX_RETRIES", 2)
    monkeypatch.setattr(cfg, "_GEMINI_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(cfg._time, "sleep", lambda *_: None)
    models = cfg._ResilientModels(_FlakyInner(fail_times=99))
    with pytest.raises(RuntimeError):
        models.generate_content(model="m", contents="hi")


def test_non_rate_limit_error_not_retried(monkeypatch):
    monkeypatch.setattr(cfg, "_GEMINI_MAX_RETRIES", 5)
    monkeypatch.setattr(cfg._time, "sleep", lambda *_: None)

    class _BadInner:
        def __init__(self):
            self.calls = 0

        def generate_content(self, *a, **k):
            self.calls += 1
            raise ValueError("400 invalid request")

    inner = _BadInner()
    models = cfg._ResilientModels(inner)
    with pytest.raises(ValueError):
        models.generate_content(model="m", contents="hi")
    assert inner.calls == 1  # not retried


def test_client_is_wrapped():
    # Config.CLIENT should expose .models.generate_content and delegate attrs.
    assert hasattr(cfg.Config.CLIENT, "models")
    assert hasattr(cfg.Config.CLIENT.models, "generate_content")
