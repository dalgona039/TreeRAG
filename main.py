import os
import time
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.routes import router
from src.api.task_routes import router as task_router
from src.config import Config
from src.middleware.security import SecurityHeadersMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

class HealthCheckCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[bool, float]] = {}
    
    def get(self, key: str) -> Optional[bool]:
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return result
        return None
    
    def set(self, key: str, value: bool) -> None:
        self.cache[key] = (value, time.time())

health_cache = HealthCheckCache(ttl_seconds=300)

async def check_api_connectivity() -> bool:
    cached = health_cache.get("api_connectivity")
    if cached is not None:
        return cached
    
    try:
        response = Config.CLIENT.models.generate_content(
            model=Config.MODEL_NAME,
            contents="ping",
            config={"max_output_tokens": 10}
        )
        result = bool(response.text)
        health_cache.set("api_connectivity", result)
        return result
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        health_cache.set("api_connectivity", False)
        return False

def check_disk_writable() -> bool:
    try:
        test_file = os.path.join(Config.INDEX_DIR, ".health_check")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"❌ Disk write check failed: {e}")
        return False

def check_indices_directory() -> bool:
    try:
        return os.path.isdir(Config.INDEX_DIR) and os.access(Config.INDEX_DIR, os.R_OK)
    except Exception as e:
        print(f"❌ Indices directory check failed: {e}")
        return False

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
    
    app.add_middleware(SecurityHeadersMiddleware)
    
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
    
    @app.get("/health")
    async def health_check():
        """Shallow health check - service is alive."""
        return {
            "status": "healthy",
            "service": "TreeRAG API",
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    @app.get("/health/deep")
    async def deep_health_check() -> Dict[str, Any]:
        """
        Deep health check - validates all critical dependencies.
        
        Results are cached for 5 minutes to avoid unnecessary API calls
        and rate limit issues with Kubernetes readiness probes.
        """
        health_status = {
            "status": "healthy",
            "service": "TreeRAG API",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {},
            "cache_info": {
                "ttl_seconds": health_cache.ttl_seconds,
                "description": "API check results cached to prevent rate limiting"
            }
        }
        
        api_ok = await check_api_connectivity()
        health_status["checks"]["gemini_api"] = {
            "status": "healthy" if api_ok else "unhealthy",
            "message": "API accessible" if api_ok else "API connection failed",
            "cached": health_cache.get("api_connectivity") is not None
        }
        
        disk_ok = check_disk_writable()
        health_status["checks"]["disk_storage"] = {
            "status": "healthy" if disk_ok else "unhealthy",
            "message": "Writable" if disk_ok else "No write permission"
        }
        
        indices_ok = check_indices_directory()
        health_status["checks"]["indices_directory"] = {
            "status": "healthy" if indices_ok else "unhealthy",
            "message": "Accessible" if indices_ok else "Directory missing"
        }
        
        all_healthy = api_ok and disk_ok and indices_ok
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    
    app.include_router(router, prefix="/api", tags=["API"])
    app.include_router(task_router, prefix="/api/tasks", tags=["Tasks"])
    
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