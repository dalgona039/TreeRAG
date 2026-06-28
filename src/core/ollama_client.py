"""
Ollama LLM client adapter compatible with the google-genai client interface.

Exposes ``OllamaClient.models.generate_content(model, contents, config, **kw)``
so it can be plugged in via ``src.config.set_client_override(OllamaClient(...))``.
All downstream callers (tree_traversal, beam_search, reasoner) continue to call
``Config.get_client(...).models.generate_content(...)`` unchanged.

Scoring prompts (tree traversal / beam search) ask the model to output JSON
("JSON만 출력하세요").  To maximise compliance we pass Ollama's ``format: "json"``
when the caller's config signals ``response_mime_type="application/json"``.

For the final answer generation the prompt is free-form Korean text; we omit the
JSON format flag so the model responds naturally.  The reasoner's
``_normalize_model_answer()`` handles both plain text and ``{"answer": "..."}``
JSON wrappers gracefully.
"""
from __future__ import annotations

import json
import urllib.error as _urlerr
import urllib.request as _urllib
from typing import Any, Optional


class OllamaResponse:
    """Minimal stand-in for google.genai GenerateContentResponse."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover
        return f"OllamaResponse(text={self.text[:80]!r})"


class OllamaModels:
    """Mimics the ``client.models`` namespace used by traversal / reasoner."""

    # Default max output tokens for answer generation.
    # Ollama's built-in default can be very small; 2048 gives room for full answers.
    DEFAULT_NUM_PREDICT = 2048

    def __init__(self, base_url: str, default_model: str, timeout: int = 300) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._generate_url = f"{self._base_url}/api/generate"

    def generate_content(
        self,
        model: Optional[str] = None,
        contents: Any = None,
        config: Any = None,
        **_kwargs: Any,
    ) -> OllamaResponse:
        used_model = self._default_model  # ignore Gemini model name passed by callers

        prompt = contents if isinstance(contents, str) else str(contents or "")

        # Use Ollama's structured JSON mode only when the caller explicitly
        # requests application/json (traversal / beam scoring prompts).
        # Free-form answer generation performs better without forced JSON.
        use_json_format = False
        if config is not None:
            mime = getattr(config, "response_mime_type", None)
            if mime == "application/json":
                use_json_format = True

        # Respect caller's max_output_tokens if provided; fall back to our default.
        num_predict = self.DEFAULT_NUM_PREDICT
        if config is not None:
            caller_max = getattr(config, "max_output_tokens", None)
            if caller_max is not None:
                num_predict = int(caller_max)

        payload: dict[str, Any] = {
            "model": used_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict},
        }
        if use_json_format:
            payload["format"] = "json"

        data = json.dumps(payload).encode()
        req = _urllib.Request(
            self._generate_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _urllib.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
        except (_urlerr.URLError, OSError) as exc:
            raise RuntimeError(f"Ollama request failed ({self._generate_url}): {exc}") from exc

        text = result.get("response", "")
        return OllamaResponse(text)

    # Provide a minimal stub for any attribute access that callers don't use
    # but that might be probed (e.g. health checks).
    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        raise AttributeError(f"OllamaModels has no attribute '{name}'")


class OllamaClient:
    """Drop-in replacement for the google-genai Resilient client.

    Usage::

        from src.core.ollama_client import OllamaClient
        from src.config import set_client_override
        set_client_override(OllamaClient(model="gemma4:12b"))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 300,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self.models = OllamaModels(base_url, model, timeout)
        self._verify_connectivity()

    def _verify_connectivity(self) -> None:
        """Raise RuntimeError early if Ollama is not reachable."""
        tags_url = f"{self._base_url}/api/tags"
        try:
            _urllib.urlopen(tags_url, timeout=5)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama not reachable at {self._base_url}: {exc}\n"
                "Make sure Ollama is running: `ollama serve`"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover
        return f"OllamaClient(base_url={self._base_url!r}, model={self._model!r})"
