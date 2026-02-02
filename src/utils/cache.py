"""
Simple in-memory cache for API responses.
Reduces Gemini API calls and speeds up repeated queries.
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any
from collections import OrderedDict


class QueryCache:
    """LRU cache with TTL for query responses."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items (default: 1 hour)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, question: str, index_files: list, 
                     use_deep_traversal: bool, max_depth: int, 
                     max_branches: int, domain_template: str, 
                     language: str, node_context: Optional[dict] = None) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "question": question.strip().lower(),
            "index_files": sorted(index_files),
            "use_deep_traversal": use_deep_traversal,
            "max_depth": max_depth,
            "max_branches": max_branches,
            "domain_template": domain_template,
            "language": language,
            "node_context": node_context
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, question: str, index_files: list, 
            use_deep_traversal: bool, max_depth: int, 
            max_branches: int, domain_template: str, 
            language: str, node_context: Optional[dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        key = self._generate_key(question, index_files, use_deep_traversal, 
                                 max_depth, max_branches, domain_template, 
                                 language, node_context)
        
        if key in self.cache:
            cached_data = self.cache[key]
            
            if time.time() - cached_data["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.cache.move_to_end(key)
            self.hits += 1
            return cached_data["response"]
        
        self.misses += 1
        return None
    
    def set(self, question: str, index_files: list, 
            use_deep_traversal: bool, max_depth: int, 
            max_branches: int, domain_template: str, 
            language: str, response: Dict[str, Any],
            node_context: Optional[dict] = None):
        """Cache a response."""
        key = self._generate_key(question, index_files, use_deep_traversal, 
                                 max_depth, max_branches, domain_template, 
                                 language, node_context)
        
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl_seconds": self.ttl_seconds
        }


_query_cache = QueryCache(max_size=100, ttl_seconds=3600)


def get_cache() -> QueryCache:
    """Get the global cache instance."""
    return _query_cache
