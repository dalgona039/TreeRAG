"""
File validation utilities with Magic Byte verification.
MIME 스푸핑 방지를 위한 실제 파일 타입 검증.
"""
import magic
from typing import Tuple, Optional
from pathlib import Path


class FileValidator:
    
    PDF_SIGNATURES = [
        b'%PDF-1.',  # PDF 1.x
        b'%PDF-2.',  # PDF 2.x
    ]
    
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/x-pdf',
    }
    
    @classmethod
    def verify_pdf(cls, file_content: bytes) -> Tuple[bool, Optional[str]]:
        """
        PDF 파일 검증 (Magic Byte 확인)
        
        Args:
            file_content: 파일 바이너리 데이터
        
        Returns:
            (검증 성공 여부, 에러 메시지)
        """
        if not file_content:
            return False, "Empty file content"
        
        file_header = file_content[:8]
        is_pdf_signature = any(
            file_header.startswith(sig) 
            for sig in cls.PDF_SIGNATURES
        )
        
        if not is_pdf_signature:
            return False, "File signature does not match PDF format"
        
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
            
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                return False, f"Invalid MIME type detected: {mime_type}"
        
        except Exception as e:
            if not is_pdf_signature:
                return False, f"MIME detection failed: {str(e)}"
        
        try:
            file_end = file_content[-1024:].decode('latin-1', errors='ignore')
            if '%%EOF' not in file_end:
                return False, "Invalid PDF structure: missing EOF marker"
        except Exception:
            pass
        
        return True, None
    
    @classmethod
    def check_file_complexity(cls, file_content: bytes, max_size: int = 50 * 1024 * 1024) -> Tuple[bool, Optional[str]]:
        """
        파일 복잡도 체크 (Decompression Bomb 방지)
        
        Args:
            file_content: 파일 바이너리
            max_size: 최대 허용 크기 (바이트)
        
        Returns:
            (안전 여부, 에러 메시지)
        """
        file_size = len(file_content)
        
        if file_size > max_size:
            return False, f"File too large: {file_size} bytes (max: {max_size})"
        
        stream_count = file_content.count(b'/FlateDecode')
        if stream_count > 10000:
            return False, f"Suspicious PDF: too many compressed streams ({stream_count})"
        
        return True, None
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        파일명 정규화
        """
        safe_name = Path(filename).name
        
        dangerous_chars = ['..', '/', '\\', '\x00', '\n', '\r']
        for char in dangerous_chars:
            safe_name = safe_name.replace(char, '_')
        
        if len(safe_name) > 255:
            name_parts = safe_name.rsplit('.', 1)
            if len(name_parts) == 2:
                name, ext = name_parts
                safe_name = name[:250] + '.' + ext
            else:
                safe_name = safe_name[:255]
        
        return safe_name


def validate_uploaded_file(
    file_content: bytes,
    filename: str,
    max_size: int = 50 * 1024 * 1024
) -> Tuple[bool, Optional[str], str]:
    """
    통합 파일 검증
    """
    safe_filename = FileValidator.sanitize_filename(filename)
    
    is_valid_pdf, pdf_error = FileValidator.verify_pdf(file_content)
    if not is_valid_pdf:
        return False, pdf_error, safe_filename
    
    is_safe_complexity, complexity_error = FileValidator.check_file_complexity(
        file_content, max_size
    )
    if not is_safe_complexity:
        return False, complexity_error, safe_filename
    
    return True, None, safe_filename
