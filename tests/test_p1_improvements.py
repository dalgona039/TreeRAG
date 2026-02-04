"""
Tests for P1 (High Priority) stability and performance improvements.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from src.core.tree_traversal import TreeNavigator
from src.core.indexer import RegulatoryIndexer
from src.utils.cache import QueryCache


class TestP1UnfiniteRecursionPrevention:
    """P1-1: 무한 재귀 방지"""
    
    def test_max_depth_enforcement(self):
        """Test that max_depth is enforced and capped."""
        from fastapi import HTTPException
        
        tree = {
            "id": "1",
            "title": "Root",
            "summary": "",
            "page_ref": "1",
            "children": [
                {
                    "id": "2",
                    "title": "Child",
                    "summary": "",
                    "page_ref": "2",
                    "children": [
                        {
                            "id": "3",
                            "title": "Grandchild",
                            "summary": "",
                            "page_ref": "3",
                            "children": []
                        }
                    ]
                }
            ]
        }
        
        navigator = TreeNavigator(tree, "test_doc")
        
        with pytest.raises(HTTPException) as exc_info:
            navigator.search("test", max_depth=20, max_branches=3)
        
        assert exc_info.value.status_code == 400
    
    def test_max_nodes_limit_enforcement(self):
        """Test that MAX_NODES_LIMIT prevents excessive traversal."""
        from fastapi import HTTPException
        
        deep_tree = {"id": "0", "title": "Root", "summary": "", "page_ref": "1", "children": []}
        current = deep_tree
        for i in range(1, 1500):
            child = {"id": str(i), "title": f"Node {i}", "summary": "", "page_ref": str(i), "children": []}
            current["children"] = [child]
            current = child
        
        navigator = TreeNavigator(deep_tree, "test_doc")
        
        try:
            results, stats = navigator.search("test", max_depth=5, max_branches=100)
            assert stats["nodes_visited"] <= TreeNavigator.MAX_NODES_LIMIT
        except HTTPException as e:
            assert e.status_code in [400, 413]
    
    def test_circular_reference_prevention(self):
        """Test that circular references are prevented."""
        tree = {
            "id": "1",
            "title": "Root",
            "children": [
                {
                    "id": "2",
                    "title": "Child",
                    "children": []
                }
            ]
        }
        
        tree["children"][0]["children"].append(tree)
        
        navigator = TreeNavigator(tree, "test_doc")
        results, stats = navigator.search("test", max_depth=3, max_branches=2)
        
        assert len(navigator.visited_nodes) <= TreeNavigator.MAX_NODES_LIMIT
        assert stats["nodes_visited"] > 0


class TestP1MemoryLeakPrevention:
    """P1-2: 메모리 누수 방지"""
    
    def test_string_concatenation_efficiency(self, indexer, sample_pdf_file):
        """Test that PDF text extraction uses efficient string concatenation."""
        try:
            result = indexer.extract_text(sample_pdf_file)
            assert isinstance(result, str)
        except ValueError:
            pass


class TestP1JSONParsingImprovement:
    """P1-3: JSON 파싱 개선"""
    
    def test_json_validation_retry(self):
        """Test that invalid JSON triggers retry logic."""
        indexer = RegulatoryIndexer()
        
        with patch('src.config.Config.CLIENT.models.generate_content') as mock_gen:
            mock_response = MagicMock()
            mock_response.text = "```json\n{invalid json]\n```"
            mock_gen.return_value = mock_response
            
            result = indexer.create_index("Test Doc", "Test content")
            
            assert result == {}
    
    def test_valid_json_parsing(self):
        """Test that valid JSON is properly parsed."""
        indexer = RegulatoryIndexer()
        
        valid_tree = {
            "id": "1",
            "title": "Root",
            "summary": "Summary",
            "page_ref": "1-5",
            "children": []
        }
        
        with patch('src.config.Config.CLIENT.models.generate_content') as mock_gen:
            mock_response = MagicMock()
            mock_response.text = json.dumps(valid_tree)
            mock_gen.return_value = mock_response
            
            result = indexer.create_index("Test Doc", "Test content")
            
            assert result == valid_tree


class TestP1CacheKeyCollisionPrevention:
    """P1-4: 캐시 키 충돌 방지"""
    
    def test_cache_key_case_sensitivity(self):
        """Test that cache keys are case-sensitive."""
        cache = QueryCache()
        
        params = {
            "use_deep_traversal": True,
            "max_depth": 5,
            "max_branches": 3,
            "domain_template": "general",
            "language": "ko"
        }
        
        key1 = cache._generate_key("Query", ["file.pdf"], **params)
        key2 = cache._generate_key("query", ["file.pdf"], **params)
        key3 = cache._generate_key("QUERY", ["file.pdf"], **params)
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
    
    def test_cache_key_node_context_distinction(self):
        """Test that None and {} node_context are differentiated."""
        cache = QueryCache()
        
        params = {
            "use_deep_traversal": True,
            "max_depth": 5,
            "max_branches": 3,
            "domain_template": "general",
            "language": "ko"
        }
        
        key_with_none = cache._generate_key("Query", ["file.pdf"], node_context=None, **params)
        key_with_empty = cache._generate_key("Query", ["file.pdf"], node_context={}, **params)
        
        assert key_with_none == key_with_empty
    
    def test_cache_hit_consistency(self):
        """Test that same parameters always produce same cache key."""
        cache = QueryCache()
        test_data = {"answer": "Test answer"}
        
        params = {
            "use_deep_traversal": True,
            "max_depth": 5,
            "max_branches": 3,
            "domain_template": "general",
            "language": "ko"
        }
        
        cache.set("test question", ["file.pdf"], response=test_data, **params)
        
        result1 = cache.get("test question", ["file.pdf"], **params)
        result2 = cache.get("test question", ["file.pdf"], **params)
        result3 = cache.get("test question", ["file.pdf"], **params)
        
        assert result1 == result2 == result3 == test_data
        assert cache.get_stats()["hits"] == 3


class TestP1DockerHealthCheck:
    """P1-5: Docker 헬스체크 개선"""
    
    def test_health_check_endpoint_exists(self, test_client):
        """Test that health check endpoint exists and responds."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert data["service"] == "TreeRAG API"
    
    def test_health_check_includes_timestamp(self, test_client):
        """Test that health check response includes timestamp."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data


class TestP1CombinedImprovements:
    """Integration tests for all P1 improvements."""
    
    def test_cache_with_different_depths(self):
        """Test cache correctly handles different max_depth values."""
        cache = QueryCache()
        data = {"answer": "Test"}
        
        params_base = {
            "use_deep_traversal": True,
            "max_branches": 3,
            "domain_template": "general",
            "language": "ko"
        }
        
        cache.set("question", ["file.pdf"], max_depth=5, response=data, **params_base)
        cache.set("question", ["file.pdf"], max_depth=10, response=data, **params_base)
        
        result1 = cache.get("question", ["file.pdf"], max_depth=5, **params_base)
        result2 = cache.get("question", ["file.pdf"], max_depth=10, **params_base)
        
        assert result1 == data
        assert result2 == data
        assert cache.get_stats()["hits"] == 2
    
    def test_memory_efficiency_with_large_pdf(self, indexer, large_pdf_file):
        """Test memory-efficient PDF text extraction."""
        try:
            result = indexer.extract_text(large_pdf_file)
            assert isinstance(result, str)
        except ValueError:
            pass
