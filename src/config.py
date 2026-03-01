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
        responseMimeType="application/json",
        automaticFunctionCalling=types.AutomaticFunctionCallingConfig(
            maximumRemoteCalls=AFC_MAX_REMOTE_CALLS
        )
    )

    @classmethod
    def get_generation_config(cls, **overrides: Any) -> types.GenerateContentConfig:
        normalized = {**overrides}

        if "response_mime_type" in normalized and "responseMimeType" not in normalized:
            normalized["responseMimeType"] = normalized.pop("response_mime_type")

        if "max_output_tokens" in normalized and "maxOutputTokens" not in normalized:
            normalized["maxOutputTokens"] = normalized.pop("max_output_tokens")

        if "automatic_function_calling" in normalized and "automaticFunctionCalling" not in normalized:
            normalized["automaticFunctionCalling"] = normalized.pop("automatic_function_calling")

        afc = normalized.get("automaticFunctionCalling")
        if isinstance(afc, dict):
            afc = {**afc}
            if "maximum_remote_calls" in afc and "maximumRemoteCalls" not in afc:
                afc["maximumRemoteCalls"] = afc.pop("maximum_remote_calls")
            afc.setdefault("maximumRemoteCalls", cls.AFC_MAX_REMOTE_CALLS)
            normalized["automaticFunctionCalling"] = types.AutomaticFunctionCallingConfig(**afc)
        elif afc is None:
            normalized["automaticFunctionCalling"] = types.AutomaticFunctionCallingConfig(
                maximumRemoteCalls=cls.AFC_MAX_REMOTE_CALLS
            )

        return types.GenerateContentConfig(
            temperature=0.0,
            responseMimeType="application/json",
            **normalized
        )

    RAW_DATA_DIR = "data/raw"
    INDEX_DIR = "data/indices"
    
    USE_DEEP_TRAVERSAL = True
    MAX_TRAVERSAL_DEPTH = 5
    MAX_BRANCHES_PER_LEVEL = 3
    FALLBACK_TO_FLAT_CONTEXT = False