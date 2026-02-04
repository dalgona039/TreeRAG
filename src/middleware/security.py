"""
Security headers middleware for FastAPI.
CSP, HSTS, X-Frame-Options 등 보안 헤더 추가.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' http://localhost:3000 http://localhost:8000",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


class ProductionSecurityHeadersMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        csp_directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self' https://api.example.com",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        return response
