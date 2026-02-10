import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.utils.redis_cache import (
    InMemoryBackend,
    HybridCache,
    REDIS_AVAILABLE
)


class TestInMemoryBackend:
    def test_set_and_get(self):
        backend = InMemoryBackend()
        backend.set("key1", {"data": "value"}, ttl=3600)
        
        result = backend.get("key1")
        assert result == {"data": "value"}
    
    def test_get_missing_key(self):
        backend = InMemoryBackend()
        result = backend.get("nonexistent")
        assert result is None
    
    def test_ttl_expiration(self):
        backend = InMemoryBackend()
        backend.set("key1", {"data": "value"}, ttl=1)
        
        assert backend.get("key1") == {"data": "value"}
        
        time.sleep(1.1)
        
        assert backend.get("key1") is None
    
    def test_lru_eviction(self):
        backend = InMemoryBackend(max_size=3)
        
        backend.set("key1", {"n": 1}, ttl=3600)
        backend.set("key2", {"n": 2}, ttl=3600)
        backend.set("key3", {"n": 3}, ttl=3600)
        
        backend.set("key4", {"n": 4}, ttl=3600)
        
        assert backend.get("key1") is None
        assert backend.get("key2") is not None
        assert backend.get("key4") is not None
    
    def test_delete(self):
        backend = InMemoryBackend()
        backend.set("key1", {"data": "value"}, ttl=3600)
        
        assert backend.delete("key1") is True
        assert backend.get("key1") is None
        assert backend.delete("key1") is False
    
    def test_clear(self):
        backend = InMemoryBackend()
        backend.set("key1", {"n": 1}, ttl=3600)
        backend.set("key2", {"n": 2}, ttl=3600)
        
        backend.clear()
        
        assert backend.get("key1") is None
        assert backend.get("key2") is None
    
    def test_stats(self):
        backend = InMemoryBackend(max_size=100)
        
        backend.set("key1", {"data": "value"}, ttl=3600)
        backend.get("key1")  # hit
        backend.get("key2")  # miss
        
        stats = backend.get_stats()
        assert stats["backend"] == "in-memory"
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestHybridCache:
    def test_init_without_redis(self):
        cache = HybridCache(redis_url=None)
        assert cache.is_redis is False
    
    def test_generate_key_consistent(self):
        cache = HybridCache()
        
        key1 = cache._generate_key(
            question="test question",
            index_files=["doc1.json", "doc2.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        key2 = cache._generate_key(
            question="test question",
            index_files=["doc2.json", "doc1.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert key1 == key2
    
    def test_generate_key_different_questions(self):
        cache = HybridCache()
        
        key1 = cache._generate_key(
            question="question 1",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        key2 = cache._generate_key(
            question="question 2",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert key1 != key2
    
    def test_get_and_set(self):
        cache = HybridCache(ttl_seconds=3600)
        
        cache.set(
            question="test question",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko",
            response={"answer": "test answer", "metadata": {}}
        )
        
        result = cache.get(
            question="test question",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert result is not None
        assert result["answer"] == "test answer"
    
    def test_cache_miss(self):
        cache = HybridCache()
        
        result = cache.get(
            question="nonexistent",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert result is None
    
    def test_clear(self):
        cache = HybridCache()
        
        cache.set(
            question="test",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko",
            response={"answer": "test"}
        )
        
        cache.clear()
        
        result = cache.get(
            question="test",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert result is None
    
    def test_stats(self):
        cache = HybridCache(ttl_seconds=3600)
        
        cache.get(
            question="miss",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        stats = cache.get_stats()
        
        assert "ttl_seconds" in stats
        assert stats["ttl_seconds"] == 3600
        assert "using_redis" in stats
        assert stats["using_redis"] is False
    
    def test_node_context_affects_key(self):
        cache = HybridCache()
        
        key1 = cache._generate_key(
            question="test",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko",
            node_context=None
        )
        
        key2 = cache._generate_key(
            question="test",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko",
            node_context={"id": "1.1", "title": "Section"}
        )
        
        assert key1 != key2


class TestRedisBackend:
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_redis_import(self):
        from src.utils.redis_cache import RedisBackend
        assert RedisBackend is not None
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_redis_connection_failure_fallback(self):
        cache = HybridCache(
            redis_url="redis://nonexistent:6379/0",
            fallback_to_memory=True
        )
        
        assert cache.is_redis is False
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_redis_url_parsing(self):
        cache = HybridCache()
        
        result = cache._parse_redis_url("redis://localhost:6379/0")
        assert result["host"] == "localhost"
        assert result["port"] == 6379
        assert result["db"] == 0
        
        result = cache._parse_redis_url("redis://:password@host:6380/1")
        assert result["host"] == "host"
        assert result["port"] == 6380
        assert result["db"] == 1
        assert result["password"] == "password"


class TestCacheIntegration:
    def test_korean_question_normalization(self):
        cache = HybridCache()
        
        cache.set(
            question="졸업 학점은 몇 학점인가요?",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko",
            response={"answer": "130학점"}
        )
        
        result = cache.get(
            question="졸업 학점은 몇 학점인가요?",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="ko"
        )
        
        assert result is not None
        assert result["answer"] == "130학점"
    
    def test_english_case_insensitive(self):
        cache = HybridCache()
        
        cache.set(
            question="What is the graduation requirement?",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="en",
            response={"answer": "130 credits"}
        )
        
        result = cache.get(
            question="what is the graduation requirement?",
            index_files=["doc.json"],
            use_deep_traversal=True,
            max_depth=5,
            max_branches=3,
            domain_template="general",
            language="en"
        )
        
        assert result is not None
        assert result["answer"] == "130 credits"
