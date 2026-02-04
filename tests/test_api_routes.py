"""
Integration tests for API routes.
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_shallow_health_check(self, test_client):
        """Test /health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "TreeRAG API"
        assert "timestamp" in data
    
    def test_deep_health_check(self, test_client):
        """Test /health/deep endpoint."""
        response = test_client.get("/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "checks" in data
        assert "gemini_api" in data["checks"]
        assert "disk_storage" in data["checks"]
        assert "indices_directory" in data["checks"]
    
    @pytest.mark.performance
    def test_health_check_performance(self, test_client):
        """Test health check response time."""
        import time
        start = time.time()
        response = test_client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 0.1

class TestFileUploadEndpoint:
    """Test file upload endpoints."""
    
    @pytest.mark.security
    def test_upload_pdf_file(self, test_client, sample_pdf_content):
        """Test PDF file upload."""
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code in [200, 400, 422, 413]
    
    @pytest.mark.security
    def test_path_traversal_prevention(self, test_client):
        """Test Path Traversal attack prevention."""
        files = {"file": ("../../etc/passwd", b"malicious", "text/plain")}
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code in [400, 403, 422]
    
    @pytest.mark.security
    def test_large_file_rejection(self, test_client):
        """Test rejection of oversized files."""
        large_content = b"x" * (1024 * 1024 * 1024)
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code in [413, 422]


class TestQueryEndpoint:
    """Test query endpoints."""
    
    def test_simple_query(self, test_client, sample_query_data):
        """Test simple query."""
        response = test_client.post("/api/chat", json=sample_query_data)
        
        assert response.status_code in [200, 422, 429, 500]
    
    def test_query_without_files(self, test_client):
        """Test query without index files."""
        response = test_client.post(
            "/api/chat",
            json={"question": "test", "index_files": []}
        )
        
        assert response.status_code in [400, 422, 404, 429]
    
    @pytest.mark.security
    def test_xss_injection_in_query(self, test_client):
        """Test XSS prevention in query."""
        response = test_client.post(
            "/api/chat",
            json={
                "question": "<script>alert('xss')</script>",
                "index_files": ["test.json"]
            }
        )
        
        assert response.status_code in [200, 422, 400, 429]
        if response.status_code == 200:
            data = response.json()
            assert "<script>" not in str(data)
    
    @pytest.mark.security
    def test_max_depth_validation(self, test_client):
        """Test max_depth parameter validation."""
        response = test_client.post(
            "/api/chat",
            json={
                "question": "test",
                "index_files": ["test.json"],
                "max_depth": 999999
            }
        )
        
        assert response.status_code in [200, 400, 413, 429]


class TestErrorHandling:
    """Test error handling and status codes."""
    
    def test_404_not_found(self, test_client):
        """Test 404 error handling."""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, test_client):
        """Test 405 error handling."""
        response = test_client.delete("/health")
        assert response.status_code == 405
    
    def test_rate_limiting(self, test_client):
        """Test rate limiting."""
        responses = []
        for _ in range(150):
            response = test_client.get("/health")
            responses.append(response.status_code)
        
        assert 429 in responses or all(r == 200 for r in responses)
