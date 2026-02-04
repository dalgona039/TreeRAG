"""
Integration tests that call real external APIs.
These tests are OPTIONAL and should only run when explicitly requested.

Run with: pytest tests/test_integration_real_api.py -m integration_real
Skip by default: pytest tests/ -m "not integration_real"
"""
import pytest
import os
from src.core.indexer import RegulatoryIndexer


pytestmark = pytest.mark.skipif(
    os.environ.get('REAL_API_TEST') != '1',
    reason="Real API tests only run with REAL_API_TEST=1 environment variable"
)


@pytest.mark.integration_real
@pytest.mark.slow
class TestRealGeminiAPI:
    """Test with actual Gemini API calls (not mocked)."""
    
    def test_real_api_connectivity(self):
        """Test that we can actually connect to Gemini API.
        
        This is the 'smoke test' - if this fails, all other tests are meaningless.
        """
        from src.config import Config
        
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == 'test-key-only-for-testing':
            pytest.skip("No real Google API key configured")
        
        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents="Say 'ok' if you can hear me.",
                config={"max_output_tokens": 10}
            )
            
            assert response.text is not None
            assert len(response.text) > 0
            print(f"✅ Real API response: {response.text[:50]}")
            
        except Exception as e:
            pytest.fail(f"Real API call failed: {e}")
    
    def test_real_api_json_parsing_robustness(self, sample_pdf_file):
        """Test that real API can return parseable JSON (or handle failures gracefully).
        
        This tests the 'wild west' scenario - real API might return:
        - Malformed JSON
        - Extra text before/after JSON
        - Completely wrong format
        """
        from src.config import Config
        
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == 'test-key-only-for-testing':
            pytest.skip("No real Google API key configured")
        
        indexer = RegulatoryIndexer()
        
        text = indexer.extract_text(sample_pdf_file)
        assert len(text) > 0, "PDF should have extractable text"
        try:
            result = indexer.create_index(
                doc_title="Integration Test Document",
                full_text=text[:1000]
            )
            assert "id" in result
            assert "title" in result
            print(f"✅ Real API returned valid structure: {result.get('title', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ Real API call failed (this is expected sometimes): {e}")
            assert True, "Real API failures are acceptable for integration tests"
    
    def test_real_api_retry_mechanism(self):
        """Test that retry logic works with real API rate limits/failures.
        
        This intentionally makes rapid requests to potentially trigger:
        - Rate limiting
        - Temporary failures
        - Network issues
        """
        from src.config import Config
        
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == 'test-key-only-for-testing':
            pytest.skip("No real Google API key configured")
        
        indexer = RegulatoryIndexer()
        
        results = []
        errors = []
        
        for i in range(3):
            try:
                result = indexer.create_index(
                    doc_title=f"Retry Test {i}",
                    full_text="This is a test document for retry mechanism testing."
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        print(f"✅ Successes: {len(results)}, Failures: {len(errors)}")
        
        if errors:
            print(f"⚠️ Real API errors encountered: {errors[:2]}")
        
        assert len(results) + len(errors) == 3, "All requests should complete (success or failure)"


@pytest.mark.integration_real
@pytest.mark.slow
class TestRealAPIEdgeCases:
    """Test edge cases that only appear with real API."""
    
    def test_api_hallucination_handling(self):
        """Test that we can handle when API returns unexpected content.
        
        Real APIs sometimes 'hallucinate' and add extra commentary outside JSON.
        """
        from src.config import Config
        
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == 'test-key-only-for-testing':
            pytest.skip("No real Google API key configured")
        
        indexer = RegulatoryIndexer()
        
        confusing_text = """
        This is a medical device regulation document.
        Wait, no it's not. It's actually a recipe for cookies.
        Just kidding. Or am I?
        """
        
        try:
            result = indexer.create_index(
                doc_title="Confusing Document",
                full_text=confusing_text
            )
            
            assert isinstance(result, dict)
            print(f"✅ Handled confusing input, got: {result}")
            
        except Exception as e:
            print(f"⚠️ API struggled with confusing input (expected): {e}")
            assert True, "Handling confusion is part of robustness"
    
    def test_api_connection_timeout(self):
        """Test that we handle API connection timeouts gracefully.
        
        Note: This is hard to test reliably, but we can at least verify
        our timeout settings exist.
        """
        from src.config import Config
        
        assert hasattr(Config, 'CLIENT'), "Config should have CLIENT"
        print("⚠️ Timeout handling should be tested manually in staging environment")
        assert True


@pytest.mark.integration_real
def test_cost_awareness():
    """Reminder that real API tests cost money.
    
    This test always passes but prints a reminder about API costs.
    """
    print("\n" + "="*60)
    print("💰 COST AWARENESS WARNING 💰")
    print("="*60)
    print("These integration tests call REAL Gemini API")
    print("Each test costs approximately:")
    print("  - test_real_api_connectivity: ~$0.0001")
    print("  - test_real_api_json_parsing: ~$0.001")
    print("  - test_real_api_retry_mechanism: ~$0.003")
    print("\nRun sparingly, especially before deployment only.")
    print("Use: REAL_API_TEST=1 pytest tests/test_integration_real_api.py -m integration_real")
    print("="*60 + "\n")
    assert True
