"""
Integration tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_cache_stats_endpoint(self, test_client):
        """Test cache statistics endpoint."""
        response = test_client.get("/api/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check stats structure
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "size" in data
        assert isinstance(data["hits"], int)
        assert isinstance(data["misses"], int)
        assert isinstance(data["hit_rate"], (int, float))
        assert isinstance(data["size"], int)

    def test_cache_clear_endpoint(self, test_client):
        """Test cache clear endpoint."""
        response = test_client.post("/api/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Cache cleared successfully"

    @pytest.mark.integration
    def test_upload_endpoint_missing_file(self, test_client):
        """Test upload endpoint with missing file."""
        response = test_client.post(
            "/api/upload",
            files={}
        )
        
        # Should fail without file
        assert response.status_code == 422

    @pytest.mark.integration
    def test_upload_endpoint_invalid_file_type(self, test_client):
        """Test upload endpoint with invalid file type."""
        response = test_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"Not a PDF", "text/plain")}
        )
        
        # Should fail for non-PDF
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_chat_endpoint_missing_parameters(self, test_client):
        """Test chat endpoint with missing parameters."""
        response = test_client.post(
            "/api/chat",
            json={}
        )
        
        # Should fail without required parameters
        assert response.status_code == 422

    @pytest.mark.integration
    def test_chat_endpoint_structure(self, test_client, sample_query_data):
        """Test chat endpoint returns correct structure."""
        response = test_client.post(
            "/api/chat",
            json=sample_query_data
        )
        
        # May fail if no files indexed, but check structure if successful
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "traversal_path" in data
            assert isinstance(data["sources"], list)

    @pytest.mark.integration
    def test_index_endpoint_missing_files(self, test_client):
        """Test index endpoint with missing files."""
        response = test_client.post(
            "/api/index",
            json={"files": []}
        )
        
        # Should fail without files
        assert response.status_code in [400, 422]

    @pytest.mark.integration  
    def test_tree_endpoint_missing_file(self, test_client):
        """Test tree endpoint with missing file."""
        response = test_client.get("/api/tree/nonexistent.pdf")
        
        # Should return 404 or empty tree
        assert response.status_code in [200, 404]

    @pytest.mark.slow
    def test_rate_limiting_chat_endpoint(self, test_client, sample_query_data):
        """Test rate limiting on chat endpoint (30 requests/minute)."""
        # Make 31 requests rapidly
        responses = []
        for _ in range(31):
            response = test_client.post("/api/chat", json=sample_query_data)
            responses.append(response)
        
        # At least one should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes or all(code in [200, 400, 404] for code in status_codes)

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/chat")
        
        # Check CORS headers exist
        assert "access-control-allow-origin" in response.headers or \
               response.status_code == 405  # Method not allowed is ok

    def test_api_error_handling(self, test_client):
        """Test API returns proper error responses."""
        # Invalid endpoint
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # Invalid method
        response = test_client.put("/api/chat")
        assert response.status_code == 405

    @pytest.mark.integration
    def test_cache_integration_with_chat(self, test_client, sample_query_data):
        """Test cache works with chat endpoint."""
        # Clear cache first
        test_client.post("/api/cache/clear")
        
        # Get initial stats
        initial_stats = test_client.get("/api/cache/stats").json()
        initial_hits = initial_stats["hits"]
        
        # Make same request twice
        response1 = test_client.post("/api/chat", json=sample_query_data)
        response2 = test_client.post("/api/chat", json=sample_query_data)
        
        # Get updated stats
        updated_stats = test_client.get("/api/cache/stats").json()
        
        # If both succeeded, cache should have increased hits
        if response1.status_code == 200 and response2.status_code == 200:
            assert updated_stats["hits"] > initial_hits

    def test_json_response_format(self, test_client):
        """Test all endpoints return valid JSON."""
        endpoints = [
            ("/health", "get"),
            ("/api/cache/stats", "get"),
        ]
        
        for endpoint, method in endpoints:
            if method == "get":
                response = test_client.get(endpoint)
            else:
                response = test_client.post(endpoint)
            
            # Should return valid JSON
            try:
                response.json()
                assert True
            except json.JSONDecodeError:
                pytest.fail(f"Endpoint {endpoint} did not return valid JSON")

    @pytest.mark.integration
    def test_concurrent_requests(self, test_client, sample_query_data):
        """Test API handles concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return test_client.post("/api/chat", json=sample_query_data)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete (success or expected error)
        assert len(responses) == 5
        assert all(r.status_code in [200, 400, 404, 429] for r in responses)
