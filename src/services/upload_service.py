from dataclasses import dataclass
from typing import Optional

from src.repositories import DocumentRepository
from src.utils.file_validator import validate_uploaded_file


@dataclass
class UploadResult:
    success: bool
    filename: Optional[str] = None
    original_filename: Optional[str] = None
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class UploadService:
    def __init__(self, document_repository: Optional[DocumentRepository] = None):
        self.document_repo = document_repository or DocumentRepository()
    
    def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> UploadResult:
        filename_validation = self.document_repo.validate_filename(filename)
        if not filename_validation.is_valid:
            return UploadResult(
                success=False,
                error_message=filename_validation.error_message
            )
        
        safe_filename: str = filename_validation.validated_filename  # type: ignore
        
        content_validation = self.document_repo.validate_content(content, content_type)
        if not content_validation.is_valid:
            return UploadResult(
                success=False,
                error_message=content_validation.error_message
            )
        
        is_valid, error_msg, validated_name = validate_uploaded_file(
            content, safe_filename, self.document_repo.MAX_FILE_SIZE
        )
        if not is_valid:
            return UploadResult(
                success=False,
                error_message=f"File validation failed: {error_msg}"
            )
        
        try:
            final_filename: str = validated_name if validated_name else safe_filename
            metadata = self.document_repo.save(
                content=content,
                original_filename=final_filename,
                generate_unique_name=True
            )
            
            return UploadResult(
                success=True,
                filename=metadata.filename,
                original_filename=metadata.original_filename,
                file_path=metadata.file_path,
                size_bytes=metadata.size_bytes
            )
        
        except ValueError as e:
            return UploadResult(
                success=False,
                error_message=str(e)
            )
        except IOError as e:
            return UploadResult(
                success=False,
                error_message=f"Failed to save file: {str(e)}"
            )
