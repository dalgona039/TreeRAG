"""
Unit tests for QueryCache.
"""
import pytest
import time
from src.utils.cache import QueryCache


class TestQueryCache:
    """Test suite for QueryCache class."""
    
    # Default test parameters matching QueryCache API
    DEFAULT_PARAMS = {
        "use_deep_traversal": True,
        "max_depth": 3,
        "max_branches": 5,
        "domain_template": "technical",
        "language": "ko"
    }

    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = QueryCache(max_size=50, ttl_seconds=1800)
        stats = cache.get_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == "0.00%"
        assert stats["size"] == 0

    def test_cache_miss_on_new_query(self):
        """Test cache returns None for uncached query."""
        cache = QueryCache()
        result = cache.get("test question", ["file.pdf"], **self.DEFAULT_PARAMS)
        
        assert result is None
        assert cache.get_stats()["misses"] == 1

    def test_cache_set_and_get(self):
        """Test cache stores and retrieves values correctly."""
        cache = QueryCache()
        test_data = {"answer": "Test answer", "sources": []}
        
        # Set value
        cache.set("test question", ["file.pdf"], **self.DEFAULT_PARAMS, response=test_data)
        
        # Get value
        result = cache.get("test question", ["file.pdf"], **self.DEFAULT_PARAMS)
        
        assert result == test_data
        assert cache.get_stats()["hits"] == 1
        assert cache.get_stats()["size"] == 1

    def test_cache_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        cache = QueryCache()
        test_data = {"answer": "Test answer"}
        
        # Set value
        cache.set("q1", ["f1.pdf"], **self.DEFAULT_PARAMS, response=test_data)
        
        # 1 hit
        cache.get("q1", ["f1.pdf"], **self.DEFAULT_PARAMS)
        
        # 2 misses
        cache.get("q2", ["f2.pdf"], **self.DEFAULT_PARAMS)
        cache.get("q3", ["f3.pdf"], **self.DEFAULT_PARAMS)
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert "33.33%" in stats["hit_rate"]

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = QueryCache(ttl_seconds=1)  # 1 second TTL
        test_data = {"answer": "Test answer"}
        
        # Set value
        cache.set("test question", ["file.pdf"], **self.DEFAULT_PARAMS, response=test_data)
        
        # Immediate get should work
        result = cache.get("test question", ["file.pdf"], **self.DEFAULT_PARAMS)
        assert result == test_data
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        result = cache.get("test question", ["file.pdf"], **self.DEFAULT_PARAMS)
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_size is exceeded."""
        cache = QueryCache(max_size=2)  # Only 2 items
        
        # Add 3 items
        cache.set("q1", ["f1.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A1"})
        cache.set("q2", ["f2.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A2"})
        cache.set("q3", ["f3.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A3"})
        
        # q1 should be evicted (oldest)
        assert cache.get("q1", ["f1.pdf"], **self.DEFAULT_PARAMS) is None
        
        # q2 and q3 should exist
        assert cache.get("q2", ["f2.pdf"], **self.DEFAULT_PARAMS) == {"answer": "A2"}
        assert cache.get("q3", ["f3.pdf"], **self.DEFAULT_PARAMS) == {"answer": "A3"}
        
        assert cache.get_stats()["size"] == 2

    def test_cache_different_parameters_different_keys(self):
        """Test different parameters generate different cache keys."""
        cache = QueryCache()
        
        # Same question, different files
        cache.set("test", ["file1.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A1"})
        cache.set("test", ["file2.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A2"})
        
        result1 = cache.get("test", ["file1.pdf"], **self.DEFAULT_PARAMS)
        result2 = cache.get("test", ["file2.pdf"], **self.DEFAULT_PARAMS)
        
        assert result1 == {"answer": "A1"}
        assert result2 == {"answer": "A2"}
        assert result1 != result2

    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        cache = QueryCache()
        
        # Add multiple items
        cache.set("q1", ["f1.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A1"})
        cache.set("q2", ["f2.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A2"})
        cache.set("q3", ["f3.pdf"], **self.DEFAULT_PARAMS, response={"answer": "A3"})
        
        assert cache.get_stats()["size"] == 3
        
        # Clear cache
        cache.clear()
        
        # All items should be gone
        assert cache.get_stats()["size"] == 0
        assert cache.get("q1", ["f1.pdf"], **self.DEFAULT_PARAMS) is None
        assert cache.get("q2", ["f2.pdf"], **self.DEFAULT_PARAMS) is None
        assert cache.get("q3", ["f3.pdf"], **self.DEFAULT_PARAMS) is None

    def test_cache_key_generation_consistency(self):
        """Test cache key is generated consistently for same input."""
        cache = QueryCache()
        test_data = {"answer": "Test"}
        
        # Set value
        cache.set("question", ["file.pdf"], **self.DEFAULT_PARAMS, response=test_data)
        
        # Get multiple times
        result1 = cache.get("question", ["file.pdf"], **self.DEFAULT_PARAMS)
        result2 = cache.get("question", ["file.pdf"], **self.DEFAULT_PARAMS)
        result3 = cache.get("question", ["file.pdf"], **self.DEFAULT_PARAMS)
        
        assert result1 == result2 == result3 == test_data
        assert cache.get_stats()["hits"] == 3

    def test_cache_with_different_languages(self):
        """Test cache differentiates by language."""
        cache = QueryCache()
        
        # Same question, different languages
        params_ko = {**self.DEFAULT_PARAMS, "language": "ko"}
        params_en = {**self.DEFAULT_PARAMS, "language": "en"}
        
        cache.set("question", ["file.pdf"], **params_ko, response={"answer": "한글 답변"})
        cache.set("question", ["file.pdf"], **params_en, response={"answer": "English answer"})
        
        result_ko = cache.get("question", ["file.pdf"], **params_ko)
        result_en = cache.get("question", ["file.pdf"], **params_en)
        
        assert result_ko == {"answer": "한글 답변"}
        assert result_en == {"answer": "English answer"}

    def test_cache_with_different_domains(self):
        """Test cache differentiates by domain template."""
        cache = QueryCache()
        
        params_medical = {**self.DEFAULT_PARAMS, "domain_template": "medical"}
        params_legal = {**self.DEFAULT_PARAMS, "domain_template": "legal"}
        
        cache.set("question", ["file.pdf"], **params_medical, response={"answer": "Medical answer"})
        cache.set("question", ["file.pdf"], **params_legal, response={"answer": "Legal answer"})
        
        result_medical = cache.get("question", ["file.pdf"], **params_medical)
        result_legal = cache.get("question", ["file.pdf"], **params_legal)
        
        assert result_medical == {"answer": "Medical answer"}
        assert result_legal == {"answer": "Legal answer"}

    def test_cache_stats_after_operations(self):
        """Test cache statistics are updated correctly after various operations."""
        cache = QueryCache(max_size=3)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Add items
        cache.set("q1", ["f1.pdf"], **self.DEFAULT_PARAMS, response={"a": "1"})
        cache.set("q2", ["f2.pdf"], **self.DEFAULT_PARAMS, response={"a": "2"})
        
        # Check size
        assert cache.get_stats()["size"] == 2
        
        # Mix hits and misses
        cache.get("q1", ["f1.pdf"], **self.DEFAULT_PARAMS)  # hit
        cache.get("q2", ["f2.pdf"], **self.DEFAULT_PARAMS)  # hit
        cache.get("q3", ["f3.pdf"], **self.DEFAULT_PARAMS)  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert "66.67%" in stats["hit_rate"]
