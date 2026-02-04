"""
Enhanced rate limiting with per-user tracking and burst protection.
"""
import time
from collections import defaultdict, deque
from typing import Dict, Tuple
from threading import Lock


class RateLimiter:
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()
    
    def is_allowed(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> Tuple[bool, Dict[str, any]]:
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            
            request_times = self.requests[key]
            while request_times and request_times[0] < cutoff:
                request_times.popleft()
            
            current_count = len(request_times)
            
            if current_count >= max_requests:
                oldest_request = request_times[0]
                retry_after = int(oldest_request + window_seconds - now) + 1
                
                return False, {
                    "current_count": current_count,
                    "limit": max_requests,
                    "window_seconds": window_seconds,
                    "retry_after": retry_after,
                    "message": f"Rate limit exceeded. Try again in {retry_after}s"
                }
            
            request_times.append(now)
            remaining = max_requests - (current_count + 1)
            
            return True, {
                "current_count": current_count + 1,
                "limit": max_requests,
                "remaining": remaining,
                "window_seconds": window_seconds,
                "reset_at": int(request_times[0] + window_seconds)
            }
    
    def get_stats(self, key: str, window_seconds: int = 60) -> Dict:
        """특정 키의 현재 통계 조회"""
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            
            request_times = self.requests[key]
            recent_requests = [t for t in request_times if t >= cutoff]
            
            return {
                "total_requests": len(recent_requests),
                "first_request": recent_requests[0] if recent_requests else None,
                "last_request": recent_requests[-1] if recent_requests else None
            }
    
    def clear(self, key: str = None):
        with self.lock:
            if key:
                self.requests.pop(key, None)
            else:
                self.requests.clear()


_rate_limiter = RateLimiter()

def get_rate_limiter() -> RateLimiter:
    return _rate_limiter
