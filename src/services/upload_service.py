"""
Upload Service - 파일 업로드 비즈니스 로직

책임:
- 파일 업로드 처리
- 검증 오케스트레이션
- 에러 처리 및 응답 생성
"""

from dataclasses import dataclass
from typing import Optional

from src.repositories import DocumentRepository
from src.utils.file_validator import validate_uploaded_file


@dataclass
class UploadResult:
    """업로드 결과"""
    success: bool
    filename: Optional[str] = None
    original_filename: Optional[str] = None
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class UploadService:
    """파일 업로드 서비스
    
    Repository Pattern을 사용하여 파일 저장을 추상화합니다.
    """
    
    def __init__(self, document_repository: Optional[DocumentRepository] = None):
        """
        Args:
            document_repository: 문서 저장소 (기본값: 새 인스턴스 생성)
        """
        self.document_repo = document_repository or DocumentRepository()
    
    def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> UploadResult:
        """파일 업로드 처리
        
        Args:
            content: 파일 바이트 데이터
            filename: 원본 파일명
            content_type: MIME 타입
            
        Returns:
            UploadResult: 업로드 결과
        """
        # 1. 파일명 검증
        filename_validation = self.document_repo.validate_filename(filename)
        if not filename_validation.is_valid:
            return UploadResult(
                success=False,
                error_message=filename_validation.error_message
            )
        
        # safe_filename is guaranteed to be non-None here since validation passed
        safe_filename: str = filename_validation.validated_filename  # type: ignore
        
        # 2. 내용 검증
        content_validation = self.document_repo.validate_content(content, content_type)
        if not content_validation.is_valid:
            return UploadResult(
                success=False,
                error_message=content_validation.error_message
            )
        
        # 3. 추가 파일 검증 (file_validator 사용)
        is_valid, error_msg, validated_name = validate_uploaded_file(
            content, safe_filename, self.document_repo.MAX_FILE_SIZE
        )
        if not is_valid:
            return UploadResult(
                success=False,
                error_message=f"File validation failed: {error_msg}"
            )
        
        # 4. 파일 저장
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
