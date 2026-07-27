"""B1 instrumentation: per-call LLM token/latency accounting.

Opt-in via ``TREERAG_TOKEN_ACCOUNTING=1`` (or ``true``/``yes``/``on``). When
disabled (the default), every hook in this module is a no-op — this is a
pure-observation addition and changes no existing behaviour or return values.

Usage at a call site::

    from src.utils.llm_accounting import stage
    with stage("traversal_dfs"):
        response = Config.get_client("traversal").models.generate_content(...)

The actual recording happens centrally where responses are received
(``src/config.py``'s ``_ResilientModels.generate_content`` for the Gemini
path, ``src/core/ollama_client.py``'s ``OllamaModels.generate_content`` for
the Ollama path, and inline in ``benchmarks/metrics/llm_judge.py``'s
``LocalJudge`` which hits Ollama's HTTP API directly) — so every LLM call
site is covered without needing to modify each one individually beyond
wrapping it in ``stage(...)``.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

ENABLED = os.getenv("TREERAG_TOKEN_ACCOUNTING", "").strip().lower() in ("1", "true", "yes", "on")

_local = threading.local()
_lock = threading.Lock()
_records: List[Dict[str, Any]] = []


def _stack() -> List[str]:
    if not hasattr(_local, "stack"):
        _local.stack = ["unknown"]
    return _local.stack


def current_stage() -> str:
    return _stack()[-1]


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Label all LLM calls made inside this block with ``name``."""
    if not ENABLED:
        yield
        return
    _stack().append(name)
    try:
        yield
    finally:
        _stack().pop()


def record(
    *,
    backend: str,
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    latency_s: float,
    stage_name: Optional[str] = None,
) -> None:
    """Record one LLM call. No-op unless accounting is enabled."""
    if not ENABLED:
        return
    entry = {
        "stage": stage_name or current_stage(),
        "backend": backend,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        ),
        "latency_s": latency_s,
        "timestamp": time.time(),
    }
    with _lock:
        _records.append(entry)


def get_records() -> List[Dict[str, Any]]:
    with _lock:
        return list(_records)


def reset() -> None:
    with _lock:
        _records.clear()


def dump(path: str) -> None:
    with _lock:
        data = list(_records)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
