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

class Config:
    CLIENT = client
    MODEL_NAME = "gemini-3.1-pro-preview"
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

        afc = normalized.get("automatic_function_calling")
        if isinstance(afc, dict):
            afc = {**afc}
            if "maximumRemoteCalls" in afc and "maximum_remote_calls" not in afc:
                afc["maximum_remote_calls"] = afc.pop("maximumRemoteCalls")
            afc.setdefault("maximum_remote_calls", cls.AFC_MAX_REMOTE_CALLS)
            normalized["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(**afc)
        elif afc is None:
            normalized["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=cls.AFC_MAX_REMOTE_CALLS
            )

        return types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            **normalized
        )

    RAW_DATA_DIR = "data/raw"
    INDEX_DIR = "data/indices"
    
    USE_DEEP_TRAVERSAL = True
    MAX_TRAVERSAL_DEPTH = 5
    MAX_BRANCHES_PER_LEVEL = 3
    FALLBACK_TO_FLAT_CONTEXT = False