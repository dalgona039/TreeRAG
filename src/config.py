import os
import logging
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Multi-key support: 케이스별로 다른 Google API key를 사용할 수 있습니다.
#
# 사용 가능한 케이스:
#   GOOGLE_API_KEY_INDEXING   - indexer.py (PDF → JSON 트리 변환)
#   GOOGLE_API_KEY_TRAVERSAL  - tree_traversal.py, beam_search.py
#   GOOGLE_API_KEY_REASONING  - reasoner.py, document_router_service.py
#   GOOGLE_API_KEY_GRAPH      - reasoning_graph.py
#   GOOGLE_API_KEY_BENCHMARK  - domain_benchmark.py
#   GOOGLE_API_KEY            - 위 케이스 키가 없을 때 fallback (필수)
#
# 원하는 케이스만 별도 키를 지정하고 나머지는 GOOGLE_API_KEY로 fallback됩니다.
# --------------------------------------------------------------------------- #

_CASE_ENV_VARS = {
    "indexing":  "GOOGLE_API_KEY_INDEXING",
    "traversal": "GOOGLE_API_KEY_TRAVERSAL",
    "reasoning": "GOOGLE_API_KEY_REASONING",
    "graph":     "GOOGLE_API_KEY_GRAPH",
    "benchmark": "GOOGLE_API_KEY_BENCHMARK",
}

_default_api_key = os.getenv("GOOGLE_API_KEY")
if not _default_api_key:
    logger.warning("GOOGLE_API_KEY not set; Gemini features will be unavailable. "
                   "Use set_client_override() to supply an alternative LLM client.")

# --------------------------------------------------------------------------- #
# Global client override: set via set_client_override() to route all
# Config.get_client() calls to an alternative backend (e.g. Ollama).
# --------------------------------------------------------------------------- #
_client_override: Any = None


def set_client_override(override_client: Any) -> None:
    """Route all Config.get_client() calls to *override_client*.

    The override object must expose a ``.models.generate_content()`` method
    compatible with the google-genai client interface.
    Call ``set_client_override(None)`` to restore normal behaviour.
    """
    global _client_override
    _client_override = override_client


class _NullClient:
    """Placeholder used when GOOGLE_API_KEY is absent and no override is set."""
    class _NullModels:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError(
                "Gemini API unavailable: GOOGLE_API_KEY not configured. "
                "Call set_client_override() with an alternative LLM client."
            )
        def __getattr__(self, name):
            raise RuntimeError("Gemini API not available: GOOGLE_API_KEY not configured")
    def __init__(self):
        self.models = self._NullModels()
    def __getattr__(self, name):
        raise RuntimeError("Gemini API not available: GOOGLE_API_KEY not configured")


def _make_client(api_key: str) -> genai.Client:
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {type(e).__name__}")
        raise ValueError("Configuration error: Failed to initialize API client") from None

# 기본 클라이언트 (fallback용)
api_key = _default_api_key
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {type(e).__name__}")
        raise ValueError("Configuration error: Failed to initialize API client") from None
else:
    client = _NullClient()  # type: ignore[assignment]

# 케이스별 클라이언트 캐시
_case_clients: dict[str, genai.Client] = {}

def _get_or_create_client(case: str) -> genai.Client:
    """케이스에 맞는 API 클라이언트를 반환. 없으면 기본 클라이언트 사용."""
    if case not in _case_clients:
        env_var = _CASE_ENV_VARS.get(case)
        key = (os.getenv(env_var) if env_var else None) or _default_api_key
        _case_clients[case] = _make_client(key)
        if env_var and os.getenv(env_var):
            logger.info(f"Using dedicated API key for case '{case}' ({env_var})")
        else:
            logger.debug(f"No dedicated key for case '{case}', using default GOOGLE_API_KEY")
    return _case_clients[case]


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


if isinstance(client, _NullClient):
    _resilient_base = client  # keep as-is; no retries needed for the null sentinel
else:
    client = _ResilientClient(client)
    _resilient_base = client

# 케이스별 resilient 클라이언트 캐시
_resilient_case_clients: dict[str, "_ResilientClient"] = {}


class Config:
    CLIENT = _resilient_base  # use _NullClient or _ResilientClient depending on API key
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

    @classmethod
    def get_client(cls, case: str = "default") -> Any:
        """케이스별 API 클라이언트 반환.

        When a global client override is active (set via ``set_client_override()``),
        that override is returned for *all* cases, enabling full backend substitution
        (e.g. local Ollama) without touching individual call sites.

        case 값:
            "indexing"  - GOOGLE_API_KEY_INDEXING 사용
            "traversal" - GOOGLE_API_KEY_TRAVERSAL 사용
            "reasoning" - GOOGLE_API_KEY_REASONING 사용
            "graph"     - GOOGLE_API_KEY_GRAPH 사용
            "benchmark" - GOOGLE_API_KEY_BENCHMARK 사용
            기타/미지정 - GOOGLE_API_KEY (기본) 사용
        """
        if _client_override is not None:
            return _client_override
        if case not in _resilient_case_clients:
            raw = _get_or_create_client(case)
            _resilient_case_clients[case] = _ResilientClient(raw)
        return _resilient_case_clients[case]

    RAW_DATA_DIR = "data/raw"
    INDEX_DIR = "data/indices"

    USE_DEEP_TRAVERSAL = True
    MAX_TRAVERSAL_DEPTH = 5
    MAX_BRANCHES_PER_LEVEL = 3
    FALLBACK_TO_FLAT_CONTEXT = False