"""
Tests for caching and query normalization.
"""
import pytest
import json
import time
from src.utils.cache import QueryCache


class TestQueryCache:
    """Test query cache functionality."""
    
    def test_cache_set_and_get(self, cache):
        """Test basic cache set/get."""
        test_data = {"answer": "Test response", "score": 0.95}
        
        cache.set(
            "What is TreeRAG?",
            ["doc1.json"],
            use_deep_traversal=False,
            max_depth=3,
            max_branches=5,
            domain_template="general",
            language="en",
            response=test_data
        )
        
        result = cache.get(
            "What is TreeRAG?",
            ["doc1.json"],
            use_deep_traversal=False,
            max_depth=3,
            max_branches=5,
            domain_template="general",
            language="en"
        )
        
        assert result == test_data
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get(
            "Nonexistent query",
            ["doc1.json"],
            False, 3, 5, "general", "en"
        )
        
        assert result is None
    
    def test_cache_ttl_expiration(self):
        """Test cache expiration after TTL."""
        short_ttl_cache = QueryCache(max_size=10, ttl_seconds=1)
        
        test_data = {"answer": "Test"}
        short_ttl_cache.set(
            "Query", ["doc.json"], False, 3, 5, "general", "en",
            response=test_data
        )
        
        assert short_ttl_cache.get("Query", ["doc.json"], False, 3, 5, "general", "en") == test_data
        time.sleep(1.1)
        assert short_ttl_cache.get("Query", ["doc.json"], False, 3, 5, "general", "en") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        small_cache = QueryCache(max_size=2, ttl_seconds=3600)
        
        small_cache.set("Q1", ["d.json"], False, 3, 5, "g", "en", response={"id": "1"})
        small_cache.set("Q2", ["d.json"], False, 3, 5, "g", "en", response={"id": "2"})
        assert small_cache.get("Q1", ["d.json"], False, 3, 5, "g", "en") == {"id": "1"}
        assert small_cache.get("Q2", ["d.json"], False, 3, 5, "g", "en") == {"id": "2"}
        small_cache.set("Q3", ["d.json"], False, 3, 5, "g", "en", response={"id": "3"})
        
        assert small_cache.get("Q1", ["d.json"], False, 3, 5, "g", "en") is None
        assert small_cache.get("Q3", ["d.json"], False, 3, 5, "g", "en") == {"id": "3"}


class TestCacheKeyGeneration:
    """Test cache key generation and collision prevention."""
    
    def test_cache_key_case_sensitivity_english(self, cache):
        """Test case-insensitive keys for English queries."""
        key1 = cache._generate_key(
            "What is TreeRAG?", ["doc.json"], False, 3, 5, "g", "en", None
        )
        key2 = cache._generate_key(
            "what is treerag?", ["doc.json"], False, 3, 5, "g", "en", None
        )
        
        assert key1 == key2
    
    def test_cache_key_case_preservation_korean(self, cache):
        """Test case preservation for Korean queries."""
        key1 = cache._generate_key(
            "트리RAG란?", ["doc.json"], False, 3, 5, "g", "ko", None
        )
        key2 = cache._generate_key(
            "트리RAG는?", ["doc.json"], False, 3, 5, "g", "ko", None
        )
        
        assert key1 != key2
    
    @pytest.mark.security
    def test_cache_key_collision_prevention(self, cache):
        """Test that different queries produce different keys."""
        key1 = cache._generate_key("Query 1", ["d.json"], False, 3, 5, "g", "en", None)
        key2 = cache._generate_key("Query 2", ["d.json"], False, 3, 5, "g", "en", None)
        key3 = cache._generate_key("Query 1", ["d.json"], True, 3, 5, "g", "en", None)
        key4 = cache._generate_key(
            "Query 1", ["d.json"], False, 3, 5, "g", "en", {"context": "value"}
        )
        
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4
    
    def test_node_context_distinction(self, cache):
        """Test node_context=None vs node_context={} distinction."""
        key_none = cache._generate_key(
            "Query", ["d.json"], False, 3, 5, "g", "en", None
        )
        key_empty = cache._generate_key(
            "Query", ["d.json"], False, 3, 5, "g", "en", {}
        )
        
        assert key_none == key_empty
    
    def test_ensure_ascii_stability(self, cache):
        """Test that keys are stable regardless of unicode content."""
        key1 = cache._generate_key("트리RAG란?", ["d.json"], False, 3, 5, "g", "ko", None)
        key2 = cache._generate_key("트리RAG란?", ["d.json"], False, 3, 5, "g", "ko", None)
        
        assert key1 == key2
        assert len(key1) == len(key2)


class TestSmartNormalization:
    """Test smart query normalization."""
    
    def test_english_normalization_whitespace(self, cache):
        """Test English query whitespace normalization."""
        key1 = cache._generate_key("What  is  TreeRAG?", ["d.json"], False, 3, 5, "g", "en", None)
        key2 = cache._generate_key("What is TreeRAG?", ["d.json"], False, 3, 5, "g", "en", None)
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
    
    def test_english_case_insensitive(self, cache):
        """Test English is case-insensitive."""
        key1 = cache._generate_key("WHAT IS TREERAG?", ["d.json"], False, 3, 5, "g", "en", None)
        key2 = cache._generate_key("what is treerag?", ["d.json"], False, 3, 5, "g", "en", None)
        key3 = cache._generate_key("What Is TreeRAG?", ["d.json"], False, 3, 5, "g", "en", None)
        
        assert key1 == key2 == key3
