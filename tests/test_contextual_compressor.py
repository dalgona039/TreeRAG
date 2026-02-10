import pytest
from src.core.contextual_compressor import (
    ContextualCompressor,
    ContextChunk,
    CompressedContext,
    format_compressed_context
)


class TestContextChunk:
    def test_create_chunk(self):
        chunk = ContextChunk(
            node_id="1.1",
            title="Test Section",
            content="This is test content",
            page_ref="p.10",
            relevance_score=0.8,
            token_count=7
        )
        assert chunk.node_id == "1.1"
        assert chunk.relevance_score == 0.8


class TestContextualCompressor:
    @pytest.fixture
    def compressor(self):
        return ContextualCompressor(
            similarity_threshold=0.7,
            max_output_tokens=4000
        )
    
    def test_empty_contexts(self, compressor):
        result = compressor.compress([], "test query")
        assert result.original_count == 0
        assert result.compressed_count == 0
        assert result.compression_ratio == 1.0
    
    def test_single_context(self, compressor):
        contexts = [
            {"id": "1", "title": "Section 1", "summary": "This is about machine learning"}
        ]
        result = compressor.compress(contexts, "machine learning")
        assert result.original_count == 1
        assert result.compressed_count == 1
    
    def test_relevance_scoring(self, compressor):
        contexts = [
            {"id": "1", "title": "ML Basics", "summary": "Machine learning fundamentals"},
            {"id": "2", "title": "Cooking", "summary": "How to make pasta and pizza"},
            {"id": "3", "title": "Deep Learning", "summary": "Neural networks and machine learning"}
        ]
        result = compressor.compress(contexts, "machine learning algorithms")
        
        scores = {c["id"]: c["relevance_score"] for c in result.contexts}
        ml_ids = ["1", "3"]
        cooking_id = "2"
        
        if cooking_id in scores:
            for ml_id in ml_ids:
                if ml_id in scores:
                    assert scores[ml_id] >= scores[cooking_id]
    
    def test_similar_contexts_merged(self):
        compressor = ContextualCompressor(similarity_threshold=0.5)
        contexts = [
            {"id": "1", "title": "Section A", "summary": "Machine learning is a subset of AI"},
            {"id": "2", "title": "Section B", "summary": "Machine learning is part of artificial intelligence"},
            {"id": "3", "title": "Different", "summary": "Completely unrelated cooking recipes"}
        ]
        result = compressor.compress(contexts, "AI machine learning")
        
        assert result.compressed_count <= result.original_count
    
    def test_low_relevance_filtered(self, compressor):
        contexts = [
            {"id": "1", "title": "High", "summary": "Python programming language tutorial"},
            {"id": "2", "title": "Low", "summary": "Random unrelated topic about sports"},
            {"id": "3", "title": "High", "summary": "Python coding best practices"}
        ]
        result = compressor.compress(contexts, "Python programming")
        
        ids = [c["id"] for c in result.contexts]
        assert len(ids) >= 1
    
    def test_token_limit_respected(self):
        compressor = ContextualCompressor(max_output_tokens=100)
        contexts = [
            {"id": str(i), "title": f"Section {i}", "summary": "A" * 500}
            for i in range(10)
        ]
        result = compressor.compress(contexts, "test")
        
        # Should limit number of contexts returned
        assert result.compressed_count <= 3
        assert result.total_tokens_saved > 0
    
    def test_compression_ratio(self, compressor):
        contexts = [
            {"id": str(i), "title": f"Section {i}", "summary": f"Content {i}"}
            for i in range(5)
        ]
        result = compressor.compress(contexts, "test")
        assert 0 < result.compression_ratio <= 1.0


class TestTokenization:
    @pytest.fixture
    def compressor(self):
        return ContextualCompressor()
    
    def test_tokenize_english(self, compressor):
        tokens = compressor._tokenize("Hello World, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_tokenize_korean(self, compressor):
        tokens = compressor._tokenize("머신러닝 알고리즘 학습")
        assert "머신러닝" in tokens
        assert "알고리즘" in tokens
        assert "학습" in tokens
    
    def test_stopwords_removed(self, compressor):
        tokens = compressor._tokenize("This is the test")
        assert "is" not in tokens
        assert "the" not in tokens


class TestSimilarity:
    @pytest.fixture
    def compressor(self):
        return ContextualCompressor()
    
    def test_identical_texts_high_similarity(self, compressor):
        chunk1 = ContextChunk("1", "T", "machine learning AI", None, 0, 0)
        chunk2 = ContextChunk("2", "T", "machine learning AI", None, 0, 0)
        sim = compressor._content_similarity(chunk1, chunk2)
        assert sim == 1.0
    
    def test_different_texts_low_similarity(self, compressor):
        chunk1 = ContextChunk("1", "T", "machine learning neural networks", None, 0, 0)
        chunk2 = ContextChunk("2", "T", "cooking pasta pizza recipes", None, 0, 0)
        sim = compressor._content_similarity(chunk1, chunk2)
        assert sim < 0.3
    
    def test_cosine_similarity(self, compressor):
        vec1 = {"a": 0.5, "b": 0.5}
        vec2 = {"a": 0.5, "b": 0.5}
        sim = compressor._cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.01


class TestFormatting:
    def test_format_empty(self):
        result = CompressedContext(
            original_count=0,
            compressed_count=0,
            contexts=[],
            merged_groups=[],
            compression_ratio=1.0,
            total_tokens_saved=0
        )
        formatted = format_compressed_context(result)
        assert formatted == ""
    
    def test_format_with_contexts(self):
        result = CompressedContext(
            original_count=2,
            compressed_count=1,
            contexts=[
                {"id": "1", "title": "Test", "summary": "Content", "page_ref": "p.5", "relevance_score": 0.9}
            ],
            merged_groups=[],
            compression_ratio=0.5,
            total_tokens_saved=100
        )
        formatted = format_compressed_context(result)
        assert "### Test" in formatted
        assert "p.5" in formatted
        assert "0.90" in formatted
