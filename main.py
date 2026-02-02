import os
from typing import List
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.routes import router
from src.config import Config

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

def initialize_directories() -> None:
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(Config.INDEX_DIR, exist_ok=True)

def create_app() -> FastAPI:
    initialize_directories()
    
    app = FastAPI(
        title="TreeRAG API",
        version="1.0.0",
        description="AI-powered regulatory document consultation system",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    app.include_router(router, prefix="/api", tags=["API"])
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )