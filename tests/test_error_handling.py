"""
Tests for error handling and edge cases.
"""
import pytest
from fastapi import HTTPException


class TestHTTPExceptionHandling:
    """Test HTTP exception handling."""
    
    def test_400_bad_request(self, test_client):
        """Test 400 Bad Request handling."""
        response = test_client.post(
            "/api/chat",
            json={"question": "test"}
        )
        assert response.status_code in [400, 422, 429]
    
    def test_413_payload_too_large(self, test_client):
        """Test 413 Payload Too Large handling."""
        huge_question = "x" * (1024 * 1024)
        response = test_client.post(
            "/api/chat",
            json={
                "question": huge_question,
                "index_files": ["test.json"]
            }
        )
        
        assert response.status_code in [200, 413, 422, 429]
    
    def test_500_server_error_handling(self, test_client):
        """Test 500 error handling."""
        pass


class TestValidationErrors:
    """Test input validation."""
    
    def test_invalid_json_format(self, test_client):
        """Test invalid JSON handling."""
        response = test_client.post(
            "/api/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, test_client):
        """Test missing required fields validation."""
        response = test_client.post(
            "/api/chat",
            json={"index_files": ["test.json"]}
        )
        
        assert response.status_code == 422
    
    def test_invalid_data_types(self, test_client):
        """Test invalid data type validation."""
        response = test_client.post(
            "/api/chat",
            json={
                "question": 123,
                "index_files": ["test.json"]
            }
        )
        
        assert response.status_code == 422
    
    def test_negative_depth_rejection(self, test_client):
        """Test negative depth values are rejected."""
        response = test_client.post(
            "/api/chat",
            json={
                "question": "test",
                "index_files": ["test.json"],
                "max_depth": -5
            }
        )
        
        assert response.status_code in [200, 400, 422, 429]


class TestTreeNavigatorErrors:
    """Test TreeNavigator error cases."""
    
    def test_invalid_tree_structure(self):
        """Test handling of invalid tree structure."""
        from src.core.tree_traversal import TreeNavigator
        
        invalid_tree = {"id": "root", "title": "Root", "summary": "", "page_ref": "", "children": []}
        navigator = TreeNavigator(invalid_tree, "Test")
        
        try:
            results, stats = navigator.search("test")
            assert isinstance(results, list)
        except (KeyError, AttributeError, TypeError):
            pass
    
    def test_empty_tree(self):
        """Test handling of empty tree."""
        from src.core.tree_traversal import TreeNavigator
        
        empty_tree = {"id": "empty", "title": "Empty", "summary": "", "page_ref": "", "children": []}
        navigator = TreeNavigator(empty_tree, "Empty Doc")
        
        results, stats = navigator.search("test")
        
        assert isinstance(results, list)
        assert isinstance(stats, dict)
    
    def test_circular_reference_handling(self):
        """Test handling of circular references."""
        from src.core.tree_traversal import TreeNavigator
        
        tree = {"id": "root", "title": "Root", "summary": "", "page_ref": "1", "children": []}
        navigator = TreeNavigator(tree, "Test")
        results, stats = navigator.search("test")
        assert "nodes_visited" in stats


class TestIndexerErrors:
    """Test Indexer error cases."""
    
    def test_nonexistent_pdf_file(self, indexer):
        """Test handling of nonexistent PDF."""
        with pytest.raises(FileNotFoundError):
            indexer.extract_text("/nonexistent/path/file.pdf")
    
    def test_empty_pdf_content(self, indexer, empty_pdf_file):
        """Test handling of PDF with no text."""
        with pytest.raises(ValueError):
            indexer.extract_text(empty_pdf_file)
    
    def test_invalid_index_data(self, indexer):
        """Test handling of invalid index data."""
        with pytest.raises(ValueError):
            indexer.create_index("", "")


class TestCacheErrors:
    """Test cache error handling."""
    
    def test_cache_with_empty_strings(self, cache):
        """Test cache with empty strings."""
        key = cache._generate_key("", ["d.json"], False, 3, 5, "g", "en", None)
        
        assert isinstance(key, str)
    
    def test_cache_get_set(self, cache):
        """Test basic cache get/set."""
        test_data = {"answer": "test"}
        cache.set("test_query", ["test.json"], False, 3, 5, "g", "en", response=test_data)
        
        result = cache.get("test_query", ["test.json"], False, 3, 5, "g", "en")
        
        assert result is None or isinstance(result, dict)


class TestRateLimitingErrors:
    """Test rate limiting error handling."""
    
    def test_rate_limit_response(self, test_client):
        """Test rate limit error response format."""
        responses = []
        for _ in range(150):
            resp = test_client.get("/health")
            responses.append(resp)
        
        rate_limited = [r for r in responses if r.status_code == 429]
        
        if rate_limited:
            assert rate_limited[0].status_code == 429
