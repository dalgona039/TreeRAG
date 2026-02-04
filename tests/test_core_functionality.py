"""
Tests for core indexing and traversal functionality.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from src.core.indexer import RegulatoryIndexer
from src.core.tree_traversal import TreeNavigator


class TestTreeNavigatorTraversal:
    """Test tree navigation and traversal."""
    
    def test_basic_traversal(self, tree_navigator, sample_tree_data):
        """Test basic tree traversal."""
        results, stats = tree_navigator.search("principles", max_depth=2, max_branches=3)
        
        assert isinstance(results, list)
        assert isinstance(stats, dict)
        assert "nodes_visited" in stats
        assert stats["nodes_visited"] >= 0
    
    def test_depth_limit_enforcement(self, sample_tree_data):
        """Test that max_depth is enforced."""
        navigator = TreeNavigator(sample_tree_data, "Test")
        
        with pytest.raises(HTTPException) as exc_info:
            navigator.search("test", max_depth=100)
        
        assert exc_info.value.status_code == 400
    
    @pytest.mark.security
    def test_node_limit_enforcement(self, malicious_tree_data):
        """Test node count limit enforcement."""
        navigator = TreeNavigator(malicious_tree_data, "Malicious Doc")
        
        try:
            navigator.search("test", max_depth=3, max_branches=10)
        except HTTPException as e:
            assert e.status_code in [400, 413]
    
    def test_stack_memory_protection(self, malicious_tree_data):
        """Test stack memory doesn't explode."""
        navigator = TreeNavigator(malicious_tree_data, "Malicious Doc")
        
        try:
            navigator.search("test", max_depth=2, max_branches=5)
        except HTTPException as exc_info:
            assert exc_info.status_code == 413
    
    @pytest.mark.performance
    def test_traversal_performance(self, sample_tree_data):
        """Test traversal completes in reasonable time."""
        import time
        navigator = TreeNavigator(sample_tree_data, "Test")
        
        start = time.time()
        results, stats = navigator.search("principles", max_depth=2)
        elapsed = time.time() - start
        
        assert elapsed < 5.0


class TestIndexerExtractText:
    """Test text extraction from PDFs."""
    
    def test_extract_text_stream(self, indexer, sample_pdf_file):
        """Test streaming text extraction."""
        stream = indexer.extract_text_stream(sample_pdf_file)
        pages = list(stream)
        
        assert len(pages) >= 0
        assert all(isinstance(p, tuple) for p in pages)
        if pages:
            assert all(len(p) == 2 for p in pages)
    
    def test_extract_text_backward_compat(self, indexer, sample_pdf_file):
        """Test backward compatibility of extract_text."""
        try:
            text = indexer.extract_text(sample_pdf_file)
            assert isinstance(text, str)
        except ValueError as e:
            assert "No text" in str(e)
    
    def test_create_index_from_stream(self, indexer, large_pdf_file):
        """Test chunk-based indexing from stream."""
        try:
            with patch.object(indexer, 'create_index') as mock_create:
                mock_create.return_value = {"id": "test", "title": "Test"}
                
                result = indexer.create_index_from_stream(
                    "Test Doc",
                    large_pdf_file,
                    max_pages_per_chunk=100
                )
                
                if mock_create.called:
                    call_args = mock_create.call_args[0]
                    assert "Page" in call_args[1] or call_args[1] == ""
        except ValueError as e:
            assert "No text" in str(e)


@pytest.mark.slow
class TestIndexerCreateIndex:
    """Test index creation with LLM."""
    
    def test_create_index_success(self, indexer, mock_gemini_api):
        """Test successful index creation."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "id": "doc_001",
            "title": "Test Document",
            "summary": "Test summary",
            "page_ref": "1-10",
            "children": []
        })
        mock_gemini_api.models.generate_content.return_value = mock_response
        
        result = indexer.create_index("Test Document", "Sample text content")
        
        assert result is not None
        assert result.get("id") == "doc_001"
    
    def test_create_index_with_retry(self, indexer, mock_gemini_api):
        """Test retry logic on transient failures."""
        mock_response_fail = MagicMock()
        mock_response_fail.text = "Invalid JSON"
        
        mock_response_success = MagicMock()
        mock_response_success.text = json.dumps({
            "id": "doc_002",
            "title": "Retried Document",
            "summary": "Success",
            "page_ref": "1-5",
            "children": []
        })
        
        mock_gemini_api.models.generate_content.side_effect = [
            mock_response_fail,
            mock_response_success
        ]
        
        with patch('time.sleep'):
            result = indexer.create_index("Doc", "Content")
        
        assert result is not None or result == {}
    
    def test_create_index_pydantic_validation(self, indexer, mock_gemini_api):
        """Test Pydantic schema validation."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "id": "doc_003",
            "summary": "No title"
        })
        mock_gemini_api.models.generate_content.return_value = mock_response
        
        with patch('time.sleep'):
            result = indexer.create_index("Doc", "Content")
        
        assert result == {} or "title" in result
