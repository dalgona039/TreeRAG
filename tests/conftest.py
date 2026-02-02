"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from main import app
    return TestClient(app)


@pytest.fixture
def sample_pdf_content():
    """Mock PDF content for testing."""
    return b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj"


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "question": "What is TreeRAG?",
        "files": ["test.pdf"],
        "settings": {
            "max_depth": 3,
            "max_branches": 5,
            "similarity_threshold": 0.7
        },
        "domain": "technical",
        "language": "ko"
    }


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = None
    cache.clear.return_value = None
    cache.get_stats.return_value = {"hits": 0, "misses": 0, "hit_rate": 0.0}
    return cache


@pytest.fixture
def sample_tree_nodes():
    """Sample tree nodes for testing."""
    return [
        {
            "id": "node_1",
            "content": "TreeRAG is a hierarchical RAG system.",
            "metadata": {"page": 1, "level": 0}
        },
        {
            "id": "node_2",
            "content": "It uses tree-based navigation for better context.",
            "metadata": {"page": 1, "level": 1, "parent": "node_1"}
        },
        {
            "id": "node_3",
            "content": "Deep traversal enables comprehensive answers.",
            "metadata": {"page": 2, "level": 1, "parent": "node_1"}
        }
    ]


@pytest.fixture
def sample_gemini_response():
    """Sample Gemini API response for testing."""
    return {
        "answer": "TreeRAG is a hierarchical RAG system that uses tree-based navigation.",
        "confidence": 0.95,
        "sources": ["node_1", "node_2"]
    }
