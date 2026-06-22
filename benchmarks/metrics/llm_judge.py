"""
LLM-as-Judge evaluator (PHASE C-2 of the KCI publication plan).

:class:`GeminiJudge` scores a RAG answer on three 0–5 axes (faithfulness,
relevance, completeness) using Gemini, and returns normalised scores in
[0.0, 1.0] plus a one-sentence rationale. JSON-parse / API failures degrade
gracefully to ``None`` scores so a single bad response never crashes a run.

The Gemini client is injectable so unit tests can supply mocked responses
without network access.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_AXES = ("faithfulness", "relevance", "completeness")


class GeminiJudge:
    """Uses Gemini to score answers on faithfulness / relevance / completeness."""

    JUDGE_PROMPT = """
You are an expert evaluator for a document QA system. Score the following answer
on three criteria, each from 0 to 5:

Question: {question}
Source Context: {context}
System Answer: {answer}
Expected Answer Hint: {expected}

Scoring criteria:
- faithfulness (0-5): 5 = fully grounded in source, 0 = fabricated
- relevance (0-5): 5 = directly answers the question, 0 = irrelevant
- completeness (0-5): 5 = covers all key points, 0 = missing core information

Respond in JSON only:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>,
  "reasoning": "<one sentence>"}}
"""

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None):
        if client is None or model is None:
            from src.config import Config

            client = client or Config.CLIENT
            model = model or Config.MODEL_NAME
        self.client = client
        self.model = model

    # ------------------------------------------------------------------ #
    def _empty_result(self, reasoning: str = "") -> Dict[str, Any]:
        result: Dict[str, Any] = {axis: None for axis in _AXES}
        result["reasoning"] = reasoning
        return result

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def score(
        self, question: str, context: str, answer: str, expected: str
    ) -> Dict[str, Any]:
        """Score one answer. Returns normalised (0–1) scores + reasoning.

        On any API or JSON-parse error, returns ``None`` for each axis score.
        """
        prompt = self.JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, expected=expected
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt
            )
            data = self._extract_json(getattr(response, "text", "") or "")
        except Exception as exc:  # network, empty, or malformed JSON
            return self._empty_result(f"judge_error: {type(exc).__name__}")

        result: Dict[str, Any] = {}
        try:
            for axis in _AXES:
                raw = float(data[axis])
                result[axis] = max(0.0, min(1.0, raw / 5.0))
        except (KeyError, TypeError, ValueError):
            return self._empty_result("missing_or_invalid_scores")

        result["reasoning"] = str(data.get("reasoning", "")).strip()
        return result

    def score_average(
        self, question: str, context: str, answer: str, expected: str
    ) -> Optional[float]:
        """Mean of the three normalised axes, or ``None`` if scoring failed."""
        scores = self.score(question, context, answer, expected)
        values = [scores[a] for a in _AXES]
        if any(v is None for v in values):
            return None
        return sum(values) / len(values)


class LocalJudge:
    """LLM-as-Judge using a local Ollama model (no API key required)."""

    JUDGE_PROMPT = GeminiJudge.JUDGE_PROMPT

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        import urllib.request as _urllib
        self.model = model
        self.url = f"{base_url}/api/generate"
        # quick connectivity check
        try:
            _urllib.urlopen(f"{base_url}/api/tags", timeout=3)
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {base_url}: {e}")

    def _empty_result(self, reasoning: str = "") -> Dict[str, Any]:
        result: Dict[str, Any] = {axis: None for axis in _AXES}
        result["reasoning"] = reasoning
        return result

    def score(self, question: str, context: str, answer: str, expected: str) -> Dict[str, Any]:
        import json as _json
        import urllib.request as _urllib
        import urllib.error as _urlerr

        prompt = self.JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, expected=expected
        )
        payload = _json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode()
        try:
            req = _urllib.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
            with _urllib.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read())
            text = data.get("response", "")
            parsed = GeminiJudge._extract_json(text)
        except (_urlerr.URLError, _json.JSONDecodeError, Exception) as exc:
            return self._empty_result(f"local_judge_error: {type(exc).__name__}")

        result: Dict[str, Any] = {}
        try:
            for axis in _AXES:
                raw = float(parsed[axis])
                result[axis] = max(0.0, min(1.0, raw / 5.0))
        except (KeyError, TypeError, ValueError):
            return self._empty_result("missing_or_invalid_scores")
        result["reasoning"] = str(parsed.get("reasoning", "")).strip()
        return result

    def score_average(self, question: str, context: str, answer: str, expected: str) -> Optional[float]:
        scores = self.score(question, context, answer, expected)
        values = [scores[a] for a in _AXES]
        if any(v is None for v in values):
            return None
        return sum(values) / len(values)
