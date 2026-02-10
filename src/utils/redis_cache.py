import hashlib
import json
import time
from typing import Optional, Dict, Any, Protocol
from abc import ABC, abstractmethod

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class InMemoryBackend(CacheBackend):
    def __init__(self, max_size: int = 100):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            data = self._cache[key]
            if time.time() < data.get("_expires_at", 0):
                self._cache.move_to_end(key)
                self._hits += 1
                return data["value"]
            else:
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = {
            "value": value,
            "_expires_at": time.time() + ttl
        }
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "in-memory",
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.2f}%" if total > 0 else "0.00%"
        }


class RedisBackend(CacheBackend):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "treerag:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed")
        
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self._key_prefix = key_prefix
        self._hits = 0
        self._misses = 0
        
        self._client.ping()
    
    def _prefixed_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._client.get(self._prefixed_key(key))
            if data:
                self._hits += 1
                return json.loads(data)
            self._misses += 1
            return None
        except (redis.RedisError, json.JSONDecodeError):
            self._misses += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        try:
            self._client.setex(
                self._prefixed_key(key),
                ttl,
                json.dumps(value, ensure_ascii=False)
            )
        except redis.RedisError:
            pass
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(self._prefixed_key(key)))
        except redis.RedisError:
            return False
    
    def clear(self) -> None:
        try:
            pattern = f"{self._key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
            self._hits = 0
            self._misses = 0
        except redis.RedisError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        try:
            info = self._client.info("memory")
            db_size = self._client.dbsize()
        except redis.RedisError:
            info = {}
            db_size = 0
        
        return {
            "backend": "redis",
            "db_size": db_size,
            "memory_used": info.get("used_memory_human", "N/A"),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.2f}%" if total > 0 else "0.00%"
        }


class HybridCache:
    CACHE_KEY_PREFIX = "query:"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_memory_size: int = 100,
        ttl_seconds: int = 3600,
        fallback_to_memory: bool = True
    ):
        self.ttl_seconds = ttl_seconds
        self.fallback_to_memory = fallback_to_memory
        self._backend: CacheBackend
        self._using_redis = False
        
        if redis_url and REDIS_AVAILABLE:
            try:
                parsed = self._parse_redis_url(redis_url)
                self._backend = RedisBackend(**parsed)
                self._using_redis = True
                print(f"✅ Connected to Redis: {redis_url}")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}")
                if fallback_to_memory:
                    print("   Falling back to in-memory cache")
                    self._backend = InMemoryBackend(max_size=max_memory_size)
                else:
                    raise
        else:
            self._backend = InMemoryBackend(max_size=max_memory_size)
    
    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        if url.startswith("redis://"):
            url = url[8:]
        
        password = None
        if "@" in url:
            auth, url = url.rsplit("@", 1)
            if ":" in auth:
                password = auth.split(":", 1)[1]
        
        host = "localhost"
        port = 6379
        db = 0
        
        if "/" in url:
            host_port, db_str = url.rsplit("/", 1)
            db = int(db_str) if db_str else 0
        else:
            host_port = url
        
        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            host = host_port
        
        return {
            "host": host,
            "port": port,
            "db": db,
            "password": password
        }
    
    def _generate_key(
        self,
        question: str,
        index_files: list,
        use_deep_traversal: bool,
        max_depth: int,
        max_branches: int,
        domain_template: str,
        language: str,
        node_context: Optional[dict] = None
    ) -> str:
        normalized_question = question.strip()
        if language.lower() in ['en', 'english']:
            normalized_question = normalized_question.lower()
        
        key_data = {
            "q": normalized_question,
            "idx": sorted(index_files),
            "deep": use_deep_traversal,
            "depth": max_depth,
            "branches": max_branches,
            "domain": domain_template,
            "lang": language,
            "ctx": node_context or {}
        }
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return self.CACHE_KEY_PREFIX + hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(
        self,
        question: str,
        index_files: list,
        use_deep_traversal: bool,
        max_depth: int,
        max_branches: int,
        domain_template: str,
        language: str,
        node_context: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        key = self._generate_key(
            question, index_files, use_deep_traversal,
            max_depth, max_branches, domain_template,
            language, node_context
        )
        return self._backend.get(key)
    
    def set(
        self,
        question: str,
        index_files: list,
        use_deep_traversal: bool,
        max_depth: int,
        max_branches: int,
        domain_template: str,
        language: str,
        response: Dict[str, Any],
        node_context: Optional[dict] = None
    ) -> None:
        key = self._generate_key(
            question, index_files, use_deep_traversal,
            max_depth, max_branches, domain_template,
            language, node_context
        )
        self._backend.set(key, response, self.ttl_seconds)
    
    def clear(self) -> None:
        self._backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self._backend.get_stats()
        stats["ttl_seconds"] = self.ttl_seconds
        stats["using_redis"] = self._using_redis
        return stats
    
    @property
    def is_redis(self) -> bool:
        return self._using_redis


_cache_instance: Optional[HybridCache] = None


def init_cache(
    redis_url: Optional[str] = None,
    max_memory_size: int = 100,
    ttl_seconds: int = 3600
) -> HybridCache:
    global _cache_instance
    _cache_instance = HybridCache(
        redis_url=redis_url,
        max_memory_size=max_memory_size,
        ttl_seconds=ttl_seconds
    )
    return _cache_instance


def get_hybrid_cache() -> HybridCache:
    global _cache_instance
    if _cache_instance is None:
        import os
        redis_url = os.getenv("REDIS_URL")
        _cache_instance = HybridCache(redis_url=redis_url)
    return _cache_instance
