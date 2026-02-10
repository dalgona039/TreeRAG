import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.config import Config


@dataclass
class IndexMetadata:
    filename: str
    document_name: str
    file_path: str
    has_tree: bool = True


@dataclass
class TreeNode:
    id: str
    title: str
    summary: Optional[str] = None
    page_ref: Optional[str] = None
    children: Optional[List["TreeNode"]] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
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
    INDEX_SUFFIX = "_index.json"
    
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or Config.INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
    
    def _get_index_filename(self, pdf_filename: str) -> str:
        return pdf_filename.replace(".pdf", self.INDEX_SUFFIX).replace(".PDF", self.INDEX_SUFFIX)
    
    def _validate_filename(self, filename: str) -> bool:
        if not filename:
            return False
        safe_filename = Path(filename).name
        if safe_filename != filename:
            return False
        if not filename.endswith(self.INDEX_SUFFIX):
            return False
        return True
    
    def exists(self, index_filename: str) -> bool:
        if not self._validate_filename(index_filename):
            return False
        file_path = os.path.join(self.index_dir, index_filename)
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def exists_for_pdf(self, pdf_filename: str) -> bool:
        index_filename = self._get_index_filename(pdf_filename)
        return self.exists(index_filename)
    
    def save(self, tree_data: Dict[str, Any], index_filename: str) -> IndexMetadata:
        if not self._validate_filename(index_filename):
            raise ValueError(f"Invalid index filename: {index_filename}")
        file_path = os.path.join(self.index_dir, index_filename)
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
        if not self.exists(index_filename):
            return None
        file_path = os.path.join(self.index_dir, index_filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def load_tree(self, index_filename: str) -> Optional[TreeNode]:
        data = self.load(index_filename)
        if not data:
            return None
        return TreeNode.from_dict(data)
    
    def get_summary(self, index_filename: str) -> Optional[str]:
        data = self.load(index_filename)
        if not data:
            return None
        return data.get("summary", data.get("title", ""))
    
    def delete(self, index_filename: str) -> bool:
        if not self.exists(index_filename):
            return False
        file_path = os.path.join(self.index_dir, index_filename)
        try:
            os.remove(file_path)
            return True
        except OSError:
            return False
    
    def list_all(self) -> List[str]:
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
        summaries = {}
        for filename in self.list_all():
            summary = self.get_summary(filename)
            if summary:
                summaries[filename] = summary[:200]
        return summaries
    
    def search_by_keyword(self, keyword: str) -> List[str]:
        keyword_lower = keyword.lower()
        matching = []
        for filename in self.list_all():
            if keyword_lower in filename.lower():
                matching.append(filename)
                continue
            summary = self.get_summary(filename)
            if summary and keyword_lower in summary.lower():
                matching.append(filename)
        return matching
