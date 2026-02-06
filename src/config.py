import os
import logging
from dotenv import load_dotenv
from google import genai

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
    MODEL_NAME = "gemini-3-pro-preview" 
    
    GENERATION_CONFIG = {
        "temperature": 0.0,
        "response_mime_type": "application/json"
    }

    RAW_DATA_DIR = "data/raw"
    INDEX_DIR = "data/indices"
    
    USE_DEEP_TRAVERSAL = True
    MAX_TRAVERSAL_DEPTH = 5
    MAX_BRANCHES_PER_LEVEL = 3
    FALLBACK_TO_FLAT_CONTEXT = False