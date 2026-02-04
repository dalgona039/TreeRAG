"""
Tests for enhanced rate limiter.
"""
import pytest
import time
from src.utils.rate_limiter import RateLimiter, get_rate_limiter


class TestRateLimiter:
    
    def test_basic_rate_limiting(self):
        limiter = RateLimiter()
        key = "test_user_1"
        
        for i in range(5):
            allowed, meta = limiter.is_allowed(key, max_requests=5, window_seconds=10)
            assert allowed is True
            assert meta["remaining"] == 4 - i
        
        allowed, meta = limiter.is_allowed(key, max_requests=5, window_seconds=10)
        assert allowed is False
        assert "retry_after" in meta
    
    def test_sliding_window(self):
        limiter = RateLimiter()
        key = "test_user_2"
        
        allowed, _ = limiter.is_allowed(key, max_requests=2, window_seconds=2)
        assert allowed is True
        
        allowed, _ = limiter.is_allowed(key, max_requests=2, window_seconds=2)
        assert allowed is True
        
        allowed, _ = limiter.is_allowed(key, max_requests=2, window_seconds=2)
        assert allowed is False
        
        time.sleep(2.1)
        allowed, _ = limiter.is_allowed(key, max_requests=2, window_seconds=2)
        assert allowed is True
    
    def test_multiple_users(self):
        limiter = RateLimiter()
        
        for _ in range(3):
            allowed, _ = limiter.is_allowed("user_a", max_requests=3, window_seconds=10)
            assert allowed is True
        
        for _ in range(3):
            allowed, _ = limiter.is_allowed("user_b", max_requests=3, window_seconds=10)
            assert allowed is True
        
        allowed, _ = limiter.is_allowed("user_a", max_requests=3, window_seconds=10)
        assert allowed is False
        
        allowed, _ = limiter.is_allowed("user_b", max_requests=3, window_seconds=10)
        assert allowed is False
    
    def test_clear_specific_user(self):
        limiter = RateLimiter()
        
        for _ in range(5):
            limiter.is_allowed("user_c", max_requests=5, window_seconds=10)
        
        allowed, _ = limiter.is_allowed("user_c", max_requests=5, window_seconds=10)
        assert allowed is False
        
        limiter.clear("user_c")
        
        allowed, _ = limiter.is_allowed("user_c", max_requests=5, window_seconds=10)
        assert allowed is True
    
    def test_get_stats(self):
        limiter = RateLimiter()
        key = "test_user_stats"
        
        for _ in range(3):
            limiter.is_allowed(key, max_requests=10, window_seconds=60)
        
        stats = limiter.get_stats(key, window_seconds=60)
        assert stats["total_requests"] == 3
        assert stats["first_request"] is not None
        assert stats["last_request"] is not None
    
    def test_singleton_instance(self):
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
        
        limiter1.is_allowed("shared_user", max_requests=1, window_seconds=10)
        allowed, _ = limiter2.is_allowed("shared_user", max_requests=1, window_seconds=10)
        assert allowed is False
