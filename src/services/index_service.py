
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.repositories import DocumentRepository, IndexRepository
from src.core.indexer import RegulatoryIndexer


@dataclass
class IndexResult:
    success: bool
    index_filename: Optional[str] = None
    status: str = "failed"  # "created", "existing", "failed"
    error_message: Optional[str] = None


class IndexService:
    def __init__(
        self,
        document_repository: Optional[DocumentRepository] = None,
        index_repository: Optional[IndexRepository] = None
    ):
        self.document_repo = document_repository or DocumentRepository()
        self.index_repo = index_repository or IndexRepository()
        self._indexer = None
    
    @property
    def indexer(self) -> RegulatoryIndexer:
        if self._indexer is None:
            self._indexer = RegulatoryIndexer()
        return self._indexer
    
    def _get_index_filename(self, pdf_filename: str) -> str:
        return pdf_filename.replace(".pdf", "_index.json").replace(".PDF", "_index.json")
    
    def create_index(self, pdf_filename: str) -> IndexResult:
        validation = self.document_repo.validate_filename(pdf_filename)
        if not validation.is_valid:
            return IndexResult(
                success=False,
                error_message=validation.error_message
            )
        
        # safe_filename is guaranteed non-None since validation passed
        safe_filename: str = validation.validated_filename  # type: ignore
        
        if not safe_filename.lower().endswith('.pdf'):
            return IndexResult(
                success=False,
                error_message="Filename must end with .pdf"
            )
        
        pdf_path = self.document_repo.get_path(safe_filename)
        if not pdf_path:
            return IndexResult(
                success=False,
                error_message=f"PDF file not found: {safe_filename}"
            )
        
        index_filename = self._get_index_filename(safe_filename)
        if self.index_repo.exists(index_filename):
            return IndexResult(
                success=True,
                index_filename=index_filename,
                status="existing"
            )
        
        try:
            text = self.indexer.extract_text(pdf_path)
            
            if not text or len(text.strip()) == 0:
                return IndexResult(
                    success=False,
                    error_message="Failed to extract text from PDF"
                )
            
            doc_title = safe_filename.replace(".pdf", "").replace(".PDF", "").replace("_", " ")
            
            tree = self.indexer.create_index(doc_title, text)
            
            if not tree:
                return IndexResult(
                    success=False,
                    error_message="Indexing failed: empty result"
                )
            
            self.index_repo.save(tree, index_filename)
            
            return IndexResult(
                success=True,
                index_filename=index_filename,
                status="created"
            )
        
        except Exception as e:
            return IndexResult(
                success=False,
                error_message=f"Indexing failed: {str(e)}"
            )
    
    def get_index(self, index_filename: str) -> Optional[Dict[str, Any]]:
        return self.index_repo.load(index_filename)
    
    def list_indices(self):
        return self.index_repo.list_all()
    
    def delete_index(self, index_filename: str) -> bool:
        return self.index_repo.delete(index_filename)
