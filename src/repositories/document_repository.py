
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.config import Config


@dataclass
class DocumentMetadata:
    filename: str
    original_filename: str
    file_path: str
    size_bytes: int


@dataclass
class DocumentValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    validated_filename: Optional[str] = None


class DocumentRepository:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.pdf'}
    ALLOWED_MIME_TYPES = {'application/pdf'}
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or Config.RAW_DATA_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def validate_filename(self, filename: str) -> DocumentValidationResult:
        if not filename:
            return DocumentValidationResult(
                is_valid=False,
                error_message="No filename provided"
            )
        
        safe_filename = Path(filename).name
        if not safe_filename or safe_filename != filename:
            return DocumentValidationResult(
                is_valid=False,
                error_message="Invalid filename: path traversal detected"
            )
        
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return DocumentValidationResult(
                is_valid=False,
                error_message=f"Only PDF files are allowed. Got: {file_ext}"
            )
        
        return DocumentValidationResult(
            is_valid=True,
            validated_filename=safe_filename
        )
    
    def validate_content(
        self, 
        content: bytes, 
        content_type: Optional[str] = None
    ) -> DocumentValidationResult:
        if len(content) > self.MAX_FILE_SIZE:
            return DocumentValidationResult(
                is_valid=False,
                error_message=f"File too large. Maximum size: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        if len(content) == 0:
            return DocumentValidationResult(
                is_valid=False,
                error_message="Empty file"
            )
        
        if content_type and content_type not in self.ALLOWED_MIME_TYPES:
            return DocumentValidationResult(
                is_valid=False,
                error_message=f"Invalid MIME type. Expected application/pdf, got {content_type}"
            )
        
        if not content.startswith(b'%PDF'):
            return DocumentValidationResult(
                is_valid=False,
                error_message="Invalid PDF file: missing PDF header"
            )
        
        return DocumentValidationResult(is_valid=True)
    
    def save(
        self, 
        content: bytes, 
        original_filename: str,
        generate_unique_name: bool = True
    ) -> DocumentMetadata:
        validation = self.validate_filename(original_filename)
        if not validation.is_valid:
            raise ValueError(validation.error_message)
        
        # safe_filename is guaranteed non-None since validation passed
        safe_filename: str = validation.validated_filename  # type: ignore
        
        content_validation = self.validate_content(content)
        if not content_validation.is_valid:
            raise ValueError(content_validation.error_message)
        
        final_filename: str
        if generate_unique_name:
            final_filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
        else:
            final_filename = safe_filename
        
        file_path = os.path.join(self.storage_dir, final_filename)
        
        abs_file_path = os.path.abspath(file_path)
        abs_storage_dir = os.path.abspath(self.storage_dir)
        if not abs_file_path.startswith(abs_storage_dir):
            raise ValueError("Invalid file path")
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        return DocumentMetadata(
            filename=final_filename,
            original_filename=original_filename,
            file_path=file_path,
            size_bytes=len(content)
        )
    
    def exists(self, filename: str) -> bool:
        safe_filename = Path(filename).name
        if safe_filename != filename:
            return False
        
        file_path = os.path.join(self.storage_dir, safe_filename)
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def get_path(self, filename: str) -> Optional[str]:
        if not self.exists(filename):
            return None
        
        safe_filename = Path(filename).name
        return os.path.join(self.storage_dir, safe_filename)
    
    def delete(self, filename: str) -> bool:
        file_path = self.get_path(filename)
        if not file_path:
            return False
        
        try:
            os.remove(file_path)
            return True
        except OSError:
            return False
    
    def list_all(self) -> List[str]:
        try:
            return [
                f for f in os.listdir(self.storage_dir)
                if f.lower().endswith('.pdf') and os.path.isfile(
                    os.path.join(self.storage_dir, f)
                )
            ]
        except OSError:
            return []
