"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import json
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from io import BytesIO
from pypdf import PdfWriter

os.environ['TESTING'] = '1'
os.environ['GOOGLE_API_KEY'] = 'test-key-only-for-testing'
os.environ['DISABLE_RATE_LIMIT'] = 'true'

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class MockGenAI:
    class Client:
        def models(self):
            return self
        
        def generate_content(self, **kwargs):
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "id": "test_doc",
                "title": "Test Document",
                "summary": "Test summary",
                "page_ref": "1-10",
                "children": []
            })
            return mock_response

sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()

from main import create_app
from src.core.indexer import RegulatoryIndexer
from src.core.tree_traversal import TreeNavigator
from src.utils.cache import QueryCache


class MockRateLimiter:
    """Mock rate limiter that always allows requests."""
    def __call__(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def limit(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def _inject_headers(self, response, *args, **kwargs):
        """Mock method for header injection (required by slowapi)."""
        return response
    
    def __getattr__(self, name):
        """Catch-all for any other slowapi methods."""
        def mock_method(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
        return mock_method


@pytest.fixture
def app():
    """Create FastAPI test application with dependency overrides."""
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    
    app_instance = create_app()
    mock_limiter = MockRateLimiter()
    app_instance.state.limiter = mock_limiter
    
    return app_instance


@pytest.fixture
def test_client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def create_test_pdf():
    """Factory fixture to create test PDF files with actual text content."""
    def _create_pdf(filename: str, tmpdir: str, pages: int = 1, with_text: bool = True) -> str:
        """Create a test PDF file with optional text content.
        
        Args:
            filename: PDF filename
            tmpdir: Temporary directory path
            pages: Number of pages to create
            with_text: If True, add Lorem Ipsum text to each page
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        pdf_path = Path(tmpdir) / filename
        
        if with_text:
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            
            lorem_ipsum = [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
                "Duis aute irure dolor in reprehenderit in voluptate velit esse.",
                "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui.",
                "Medical device regulations require comprehensive documentation and testing.",
                "Clinical trials must follow Good Clinical Practice (GCP) guidelines.",
                "Risk management is essential for medical device development.",
                "ISO 13485 certification ensures quality management systems.",
                "Post-market surveillance monitors device performance and safety."
            ]
            
            for page_num in range(pages):
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, 750, f"Test Document - Page {page_num + 1}")
                c.setFont("Helvetica", 12)
                y_position = 700
                for i, line in enumerate(lorem_ipsum):
                    if y_position < 100:
                        break
                    c.drawString(100, y_position, line)
                    y_position -= 20
                c.setFont("Helvetica", 10)
                c.drawString(300, 50, f"Page {page_num + 1} of {pages}")
                
                c.showPage()
            
            c.save()
        else:
            pdf_writer = PdfWriter()
            for i in range(pages):
                pdf_writer.add_blank_page(width=612, height=792)
            
            with open(pdf_path, 'wb') as f:
                pdf_writer.write(f)
        
        return str(pdf_path)
    
    return _create_pdf


@pytest.fixture
def sample_pdf_file(temp_dir, create_test_pdf):
    """Create a sample test PDF file with actual text content."""
    return create_test_pdf("test.pdf", temp_dir, pages=3, with_text=True)


@pytest.fixture
def large_pdf_file(temp_dir, create_test_pdf):
    """Create a large test PDF file (200 pages) with actual text."""
    return create_test_pdf("large.pdf", temp_dir, pages=200, with_text=True)


@pytest.fixture
def empty_pdf_file(temp_dir, create_test_pdf):
    """Create an empty test PDF file (blank pages for edge case testing)."""
    return create_test_pdf("empty.pdf", temp_dir, pages=1, with_text=False)


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
def sample_tree_data() -> dict:
    """Create sample tree structure for testing."""
    return {
        "id": "doc_001",
        "title": "Sample Regulatory Document",
        "summary": "A sample regulatory document for testing",
        "page_ref": "1-10",
        "children": [
            {
                "id": "chapter_001",
                "title": "Chapter 1: Introduction",
                "summary": "Overview of regulations",
                "page_ref": "1-5",
                "children": [
                    {
                        "id": "section_001",
                        "title": "Section 1.1: Basic Principles",
                        "summary": "Fundamental principles",
                        "page_ref": "1-2",
                        "children": []
                    },
                    {
                        "id": "section_002",
                        "title": "Section 1.2: Definitions",
                        "summary": "Key definitions",
                        "page_ref": "3-5",
                        "children": []
                    }
                ]
            },
            {
                "id": "chapter_002",
                "title": "Chapter 2: Requirements",
                "summary": "Detailed requirements",
                "page_ref": "6-10",
                "children": [
                    {
                        "id": "section_003",
                        "title": "Section 2.1: Mandatory Requirements",
                        "summary": "Must-have requirements",
                        "page_ref": "6-8",
                        "children": []
                    }
                ]
            }
        ]
    }


@pytest.fixture
def malicious_tree_data() -> dict:
    """Create deeply nested/wide tree for DoS testing."""
    return {
        "id": "malicious",
        "title": "Malicious Document",
        "summary": "Wide tree for testing limits",
        "page_ref": "1",
        "children": [
            {
                "id": f"child_{i}",
                "title": f"Child {i}",
                "summary": f"Child {i} summary",
                "page_ref": f"{i}",
                "children": [
                    {
                        "id": f"grandchild_{i}_{j}",
                        "title": f"Grandchild {i}.{j}",
                        "summary": f"Grandchild {i}.{j}",
                        "page_ref": f"{i}.{j}",
                        "children": []
                    }
                    for j in range(10)
                ]
            }
            for i in range(50)
        ]
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
def cache():
    """Create query cache instance."""
    return QueryCache(max_size=100, ttl_seconds=3600)


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
def tree_navigator(sample_tree_data):
    """Create TreeNavigator instance."""
    return TreeNavigator(sample_tree_data, "Sample Document")


@pytest.fixture
def indexer():
    """Create RegulatoryIndexer instance."""
    return RegulatoryIndexer()


@pytest.fixture
def sample_gemini_response():
    """Sample Gemini API response for testing."""
    return {
        "answer": "TreeRAG is a hierarchical RAG system that uses tree-based navigation.",
        "confidence": 0.95,
        "sources": ["node_1", "node_2"]
    }


@pytest.fixture
def mock_gemini_api():
    """Mock Gemini API responses."""
    with patch('src.config.Config.CLIENT') as mock_client:
        yield mock_client


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
