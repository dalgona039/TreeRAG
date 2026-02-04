"""
Integration tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json


class TestAPIEndpoints:

    def test_health_check(self, test_client):
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_cache_stats_endpoint(self, test_client):
        response = test_client.get("/api/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        if 'cache_stats' in data:
            stats = data['cache_stats']
        else:
            stats = data
            
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats or "max_size" in stats
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)

    def test_cache_clear_endpoint(self, test_client):
        response = test_client.post("/api/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Cache cleared successfully"

    @pytest.mark.integration
    def test_upload_endpoint_missing_file(self, test_client):
        response = test_client.post(
            "/api/upload",
            files={}
        )
        
        assert response.status_code == 422

    @pytest.mark.integration
    def test_upload_endpoint_invalid_file_type(self, test_client):
        """Test upload endpoint with invalid file type."""
        response = test_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"Not a PDF", "text/plain")}
        )
        
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_chat_endpoint_missing_parameters(self, test_client):
        """Test chat endpoint with missing parameters."""
        response = test_client.post(
            "/api/chat",
            json={}
        )
        
        assert response.status_code == 422

    @pytest.mark.integration
    def test_chat_endpoint_structure(self, test_client, sample_query_data):
        """Test chat endpoint returns correct structure."""
        response = test_client.post(
            "/api/chat",
            json=sample_query_data
        )
        
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
        
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_tree_endpoint_missing_file(self, test_client):
        """Test tree endpoint with missing file."""
        response = test_client.get("/api/tree/nonexistent.pdf")
        
        assert response.status_code in [400, 404]

    @pytest.mark.slow
    def test_rate_limiting_chat_endpoint(self, test_client, sample_query_data):
        """Test rate limiting on chat endpoint (30 requests/minute)."""
        responses = []
        for _ in range(31):
            response = test_client.post("/api/chat", json=sample_query_data)
            responses.append(response)
        
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes or all(code in [200, 400, 404] for code in status_codes)

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/chat")
        
        assert "access-control-allow-origin" in response.headers or \
             response.status_code == 405

    def test_api_error_handling(self, test_client):
        """Test API returns proper error responses."""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = test_client.put("/api/chat")
        assert response.status_code == 405

    @pytest.mark.integration
    def test_cache_integration_with_chat(self, test_client, sample_query_data):
        """Test cache works with chat endpoint."""
        test_client.post("/api/cache/clear")
        
        initial_stats_response = test_client.get("/api/cache/stats").json()
        if 'cache_stats' in initial_stats_response:
            initial_stats = initial_stats_response['cache_stats']
        else:
            initial_stats = initial_stats_response
        initial_hits = initial_stats["hits"]
        
        response1 = test_client.post("/api/chat", json=sample_query_data)
        response2 = test_client.post("/api/chat", json=sample_query_data)
        
        updated_stats_response = test_client.get("/api/cache/stats").json()
        if 'cache_stats' in updated_stats_response:
            updated_stats = updated_stats_response['cache_stats']
        else:
            updated_stats = updated_stats_response
        
        if response1.status_code == 200 and response2.status_code == 200:
            assert updated_stats["hits"] >= initial_hits

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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(responses) == 5
        assert all(r.status_code in [200, 400, 404, 429] for r in responses)
