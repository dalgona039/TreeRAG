"""Promptfoo custom provider for TreeRAG /chat endpoint.

This provider allows promptfoo eval/redteam runs to call the local TreeRAG API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


DEFAULT_API_URL = "http://localhost:8000/api/chat"
DEFAULT_TIMEOUT_SECONDS = 60


def _normalize_index_filenames(value: Any) -> list[str] | None:
    """Normalize optional index_filenames var into a list of strings."""
    if value is None:
        return None

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        normalized = [p for p in parts if p]
        return normalized or None

    return [str(value).strip()] if str(value).strip() else None


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point used by promptfoo for custom Python providers."""
    api_url = os.getenv("TREERAG_API_URL", DEFAULT_API_URL)
    timeout = int(os.getenv("TREERAG_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))

    vars_map = (context or {}).get("vars", {}) or {}

    payload: Dict[str, Any] = {
        "question": prompt,
        "enable_comparison": bool(vars_map.get("enable_comparison", True)),
        "domain_template": str(vars_map.get("domain_template", "general")),
        "language": str(vars_map.get("language", "auto")),
    }

    index_filenames = _normalize_index_filenames(vars_map.get("index_filenames"))
    if index_filenames:
        payload["index_filenames"] = index_filenames

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        warning = data.get("hallucination_warning")

        return {
            "output": answer,
            "metadata": {
                "citations": citations,
                "hallucination_warning": warning,
                "traversal_info": data.get("traversal_info"),
            },
        }

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "error": f"TreeRAG API HTTP {exc.code}: {error_body}",
            "output": "",
        }
    except urllib.error.URLError as exc:
        return {
            "error": f"TreeRAG API connection failed: {exc.reason}",
            "output": "",
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "error": f"Unexpected provider error: {exc}",
            "output": "",
        }
