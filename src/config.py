import os
import logging
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("Missing required environment variable for API authentication")
    raise ValueError("Configuration error: Missing required environment variable")

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    logger.error(f"Failed to initialize API client: {type(e).__name__}")
    raise ValueError("Configuration error: Failed to initialize API client") from None


# --------------------------------------------------------------------------- #
# Resilient client wrapper: rate-limit throttle + 429/quota retry-with-backoff.
#
# Free-tier Gemini keys are limited to a few requests/minute. A single TreeRAG
# query fires many calls (beam node scoring + answer generation), so without
# this wrapper the run hits 429 RESOURCE_EXHAUSTED and produces empty answers.
#
# Tunable via env (no behaviour change unless set):
#   GEMINI_MIN_INTERVAL_S : minimum seconds between calls (default 0 = off).
#                           Set to 13 to stay under a 5 requests/minute quota.
#   GEMINI_MAX_RETRIES    : retries on 429/quota errors (default 5).
# --------------------------------------------------------------------------- #
import time as _time
import re as _re
import random as _random

_GEMINI_MIN_INTERVAL_S = float(os.getenv("GEMINI_MIN_INTERVAL_S", "0") or 0)
_GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5") or 5)
_last_call = [0.0]


def _is_rate_limit_error(msg: str) -> bool:
    low = (msg or "").lower()
    return (
        "429" in msg
        or "resource_exhausted" in low
        or "quota" in low
        or "rate limit" in low
        or "ratelimit" in low
    )


def _retry_delay_seconds(msg: str):
    """Parse a server-suggested retry delay (e.g. retryDelay: '30s') if present."""
    m = _re.search(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", msg or "", _re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


class _ResilientModels:
    def __init__(self, inner):
        self._inner = inner

    def generate_content(self, *args, **kwargs):
        if _GEMINI_MIN_INTERVAL_S > 0:
            wait = _GEMINI_MIN_INTERVAL_S - (_time.time() - _last_call[0])
            if wait > 0:
                _time.sleep(wait)
        backoff = 2.0
        for attempt in range(_GEMINI_MAX_RETRIES + 1):
            try:
                resp = self._inner.generate_content(*args, **kwargs)
                _last_call[0] = _time.time()
                return resp
            except Exception as exc:
                msg = str(exc)
                if not _is_rate_limit_error(msg) or attempt == _GEMINI_MAX_RETRIES:
                    raise
                delay = _retry_delay_seconds(msg) or backoff
                logger.warning(
                    "Gemini rate-limited (attempt %d/%d); sleeping %.1fs",
                    attempt + 1, _GEMINI_MAX_RETRIES, delay,
                )
                _time.sleep(delay + _random.uniform(0, 1.0))
                backoff = min(backoff * 2, 60.0)
                _last_call[0] = _time.time()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ResilientClient:
    def __init__(self, inner):
        self._inner = inner
        self.models = _ResilientModels(inner.models)

    def __getattr__(self, name):
        return getattr(self._inner, name)


client = _ResilientClient(client)


class Config:
    CLIENT = client
    MODEL_NAME = "gemini-2.5-flash"
    AFC_MAX_REMOTE_CALLS = 50
    
    GENERATION_CONFIG = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=AFC_MAX_REMOTE_CALLS
        )
    )

    @classmethod
    def get_generation_config(cls, **overrides: Any) -> types.GenerateContentConfig:
        normalized = {**overrides}

        if "responseMimeType" in normalized and "response_mime_type" not in normalized:
            normalized["response_mime_type"] = normalized.pop("responseMimeType")

        if "maxOutputTokens" in normalized and "max_output_tokens" not in normalized:
            normalized["max_output_tokens"] = normalized.pop("maxOutputTokens")

        if "automaticFunctionCalling" in normalized and "automatic_function_calling" not in normalized:
            normalized["automatic_function_calling"] = normalized.pop("automaticFunctionCalling")

        base_config: dict[str, Any] = {
            "temperature": cls.GENERATION_CONFIG.temperature if cls.GENERATION_CONFIG.temperature is not None else 0.0,
            "response_mime_type": cls.GENERATION_CONFIG.response_mime_type or "application/json",
            "automatic_function_calling": cls.GENERATION_CONFIG.automatic_function_calling
            or types.AutomaticFunctionCallingConfig(maximum_remote_calls=cls.AFC_MAX_REMOTE_CALLS),
        }
        base_config.update(normalized)

        afc = base_config.get("automatic_function_calling")
        if isinstance(afc, dict):
            afc = {**afc}
            if "maximumRemoteCalls" in afc and "maximum_remote_calls" not in afc:
                afc["maximum_remote_calls"] = afc.pop("maximumRemoteCalls")
            afc.setdefault("maximum_remote_calls", cls.AFC_MAX_REMOTE_CALLS)
            base_config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(**afc)
        elif afc is None:
            base_config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=cls.AFC_MAX_REMOTE_CALLS
            )

        return types.GenerateContentConfig(**base_config)

    RAW_DATA_DIR = "data/raw"
    INDEX_DIR = "data/indices"
    
    USE_DEEP_TRAVERSAL = True
    MAX_TRAVERSAL_DEPTH = 5
    MAX_BRANCHES_PER_LEVEL = 3
    FALLBACK_TO_FLAT_CONTEXT = False