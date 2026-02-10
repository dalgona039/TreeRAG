
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.config import Config


@dataclass
class IndexMetadata:
    """인덱스 메타데이터"""
    filename: str
    document_name: str
    file_path: str
    has_tree: bool = True


@dataclass
class TreeNode:
    """트리 노드 데이터"""
    id: str
    title: str
    summary: Optional[str] = None
    page_ref: Optional[str] = None
    children: Optional[List["TreeNode"]] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        """딕셔너리에서 TreeNode 생성"""
        children = None
        if "children" in data and data["children"]:
            children = [cls.from_dict(child) for child in data["children"]]
        
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            summary=data.get("summary"),
            page_ref=data.get("page_ref"),
            children=children
        )
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        result: dict = {
            "id": self.id,
            "title": self.title
        }
        if self.summary:
            result["summary"] = self.summary
        if self.page_ref:
            result["page_ref"] = self.page_ref
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


class IndexRepository:
    """인덱스 파일 저장소
    
    단일 책임: Index JSON 파일 CRUD 작업
    """
    
    INDEX_SUFFIX = "_index.json"
    
    def __init__(self, index_dir: Optional[str] = None):
        """
        Args:
            index_dir: 인덱스 저장 디렉토리 (기본값: Config.INDEX_DIR)
        """
        self.index_dir = index_dir or Config.INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
    
    def _get_index_filename(self, pdf_filename: str) -> str:
        """PDF 파일명에서 인덱스 파일명 생성
        
        Args:
            pdf_filename: PDF 파일명
            
        Returns:
            str: 인덱스 파일명
        """
        return pdf_filename.replace(".pdf", self.INDEX_SUFFIX).replace(".PDF", self.INDEX_SUFFIX)
    
    def _validate_filename(self, filename: str) -> bool:
        """파일명 검증
        
        Args:
            filename: 검증할 파일명
            
        Returns:
            bool: 유효 여부
        """
        if not filename:
            return False
        
        # Path traversal 방지
        safe_filename = Path(filename).name
        if safe_filename != filename:
            return False
        
        # 인덱스 파일 형식 확인
        if not filename.endswith(self.INDEX_SUFFIX):
            return False
        
        return True
    
    def exists(self, index_filename: str) -> bool:
        """인덱스 존재 여부 확인
        
        Args:
            index_filename: 인덱스 파일명
            
        Returns:
            bool: 존재 여부
        """
        if not self._validate_filename(index_filename):
            return False
        
        file_path = os.path.join(self.index_dir, index_filename)
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def exists_for_pdf(self, pdf_filename: str) -> bool:
        """PDF에 대한 인덱스 존재 여부 확인
        
        Args:
            pdf_filename: PDF 파일명
            
        Returns:
            bool: 존재 여부
        """
        index_filename = self._get_index_filename(pdf_filename)
        return self.exists(index_filename)
    
    def save(self, tree_data: Dict[str, Any], index_filename: str) -> IndexMetadata:
        """인덱스 저장
        
        Args:
            tree_data: 트리 구조 데이터
            index_filename: 인덱스 파일명
            
        Returns:
            IndexMetadata: 저장된 인덱스 메타데이터
            
        Raises:
            ValueError: 유효하지 않은 파일명
            IOError: 파일 저장 실패
        """
        if not self._validate_filename(index_filename):
            raise ValueError(f"Invalid index filename: {index_filename}")
        
        file_path = os.path.join(self.index_dir, index_filename)
        
        # 경로 검증
        abs_file_path = os.path.abspath(file_path)
        abs_index_dir = os.path.abspath(self.index_dir)
        if not abs_file_path.startswith(abs_index_dir):
            raise ValueError("Invalid file path")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        
        document_name = index_filename.replace(self.INDEX_SUFFIX, "")
        
        return IndexMetadata(
            filename=index_filename,
            document_name=document_name,
            file_path=file_path,
            has_tree=True
        )
    
    def load(self, index_filename: str) -> Optional[Dict[str, Any]]:
        """인덱스 로드
        
        Args:
            index_filename: 인덱스 파일명
            
        Returns:
            Optional[Dict[str, Any]]: 트리 데이터 (없으면 None)
        """
        if not self.exists(index_filename):
            return None
        
        file_path = os.path.join(self.index_dir, index_filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def load_tree(self, index_filename: str) -> Optional[TreeNode]:
        """인덱스를 TreeNode 객체로 로드
        
        Args:
            index_filename: 인덱스 파일명
            
        Returns:
            Optional[TreeNode]: 트리 루트 노드 (없으면 None)
        """
        data = self.load(index_filename)
        if not data:
            return None
        
        return TreeNode.from_dict(data)
    
    def get_summary(self, index_filename: str) -> Optional[str]:
        """인덱스의 요약 반환
        
        Args:
            index_filename: 인덱스 파일명
            
        Returns:
            Optional[str]: 요약 텍스트
        """
        data = self.load(index_filename)
        if not data:
            return None
        
        return data.get("summary", data.get("title", ""))
    
    def delete(self, index_filename: str) -> bool:
        """인덱스 삭제
        
        Args:
            index_filename: 인덱스 파일명
            
        Returns:
            bool: 삭제 성공 여부
        """
        if not self.exists(index_filename):
            return False
        
        file_path = os.path.join(self.index_dir, index_filename)
        
        try:
            os.remove(file_path)
            return True
        except OSError:
            return False
    
    def list_all(self) -> List[str]:
        """모든 인덱스 파일 목록 반환
        
        Returns:
            List[str]: 인덱스 파일명 목록
        """
        try:
            return [
                f for f in os.listdir(self.index_dir)
                if f.endswith(self.INDEX_SUFFIX) and os.path.isfile(
                    os.path.join(self.index_dir, f)
                )
            ]
        except OSError:
            return []
    
    def get_document_summaries(self) -> Dict[str, str]:
        """모든 문서의 요약 반환
        
        Returns:
            Dict[str, str]: {인덱스 파일명: 요약} 딕셔너리
        """
        summaries = {}
        for filename in self.list_all():
            summary = self.get_summary(filename)
            if summary:
                summaries[filename] = summary[:200]  # 첫 200자만
        return summaries
    
    def search_by_keyword(self, keyword: str) -> List[str]:
        """키워드로 인덱스 검색
        
        Args:
            keyword: 검색 키워드
            
        Returns:
            List[str]: 매칭되는 인덱스 파일명 목록
        """
        keyword_lower = keyword.lower()
        matching = []
        
        for filename in self.list_all():
            # 파일명에서 검색
            if keyword_lower in filename.lower():
                matching.append(filename)
                continue
            
            # 요약에서 검색
            summary = self.get_summary(filename)
            if summary and keyword_lower in summary.lower():
                matching.append(filename)
        
        return matching
