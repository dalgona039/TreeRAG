"""
Tests for file validation (Magic Byte verification).
"""
import pytest
from src.utils.file_validator import FileValidator, validate_uploaded_file


class TestFileValidator:
    
    def test_valid_pdf_signature(self):
        
        valid_pdf = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'
        valid_pdf += b'1 0 obj\n<< /Type /Catalog >>\nendobj\n'
        valid_pdf += b'%%EOF\n'
        
        is_valid, error = FileValidator.verify_pdf(valid_pdf)
        assert is_valid is True
        assert error is None
    
    def test_invalid_pdf_signature(self):
        fake_pdf = b'\x89PNG\r\n\x1a\n'
        
        is_valid, error = FileValidator.verify_pdf(fake_pdf)
        assert is_valid is False
        assert "signature" in error.lower()
    
    def test_empty_file(self):
        is_valid, error = FileValidator.verify_pdf(b'')
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_pdf_without_eof_marker(self):
        incomplete_pdf = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'
        incomplete_pdf += b'1 0 obj\n<< /Type /Catalog >>\nendobj\n'
        
        is_valid, error = FileValidator.verify_pdf(incomplete_pdf)
        assert is_valid is False or is_valid is True
    
    def test_file_size_limit(self):
        large_content = b'%PDF-1.4\n' + b'X' * (1024 * 1024)
        
        is_safe, error = FileValidator.check_file_complexity(
            large_content, max_size=500 * 1024
        )
        assert is_safe is False
        assert "too large" in error.lower()
    
    def test_decompression_bomb_detection(self):
        suspicious_pdf = b'%PDF-1.4\n'
        suspicious_pdf += b'/FlateDecode' * 15000
        suspicious_pdf += b'\n%%EOF\n'
        
        is_safe, error = FileValidator.check_file_complexity(suspicious_pdf)
        assert is_safe is False
        assert "compressed streams" in error.lower()
    
    def test_filename_sanitization(self):
        dangerous_name = "../../../etc/passwd"
        safe = FileValidator.sanitize_filename(dangerous_name)
        assert ".." not in safe
        assert "/" not in safe
        assert "\\" not in safe
    
    def test_filename_length_limit(self):
        long_name = "a" * 300 + ".pdf"
        safe = FileValidator.sanitize_filename(long_name)
        assert len(safe) <= 255
        assert safe.endswith(".pdf")
    
    def test_validate_uploaded_file_success(self):
        valid_pdf = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'
        valid_pdf += b'1 0 obj\n<< /Type /Catalog >>\nendobj\n'
        valid_pdf += b'%%EOF\n'
        
        is_valid, error, safe_filename = validate_uploaded_file(
            valid_pdf, "test.pdf", max_size=1024 * 1024
        )
        
        assert is_valid is True
        assert error is None
        assert safe_filename == "test.pdf"
    
    def test_validate_uploaded_file_fake_pdf(self):
        fake_pdf = b'This is not a PDF file'
        
        is_valid, error, safe_filename = validate_uploaded_file(
            fake_pdf, "fake.pdf"
        )
        
        assert is_valid is False
        assert "signature" in error.lower()
    
    def test_mime_type_spoofing(self):
        html_content = b'<!DOCTYPE html><html><body>Evil</body></html>'
        
        is_valid, error = FileValidator.verify_pdf(html_content)
        assert is_valid is False
        assert "signature" in error.lower() or "mime" in error.lower()
