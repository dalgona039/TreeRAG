from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IndexRequest(BaseModel):
    filename: str

class ChatRequest(BaseModel):
    question: str
    index_filenames: Optional[List[str]] = None
    enable_comparison: bool = True
    node_context: Optional[Dict[str, Any]] = None
    use_deep_traversal: Optional[bool] = None
    max_depth: Optional[int] = None
    max_branches: Optional[int] = None
    domain_template: Optional[str] = "general"  
    language: Optional[str] = "ko" 

class ComparisonResult(BaseModel):
    has_comparison: bool
    documents_compared: List[str]
    commonalities: Optional[str] = None
    differences: Optional[str] = None

class TraversalInfo(BaseModel):
    used_deep_traversal: bool
    nodes_visited: List[str]
    nodes_selected: List[Dict[str, Any]]
    max_depth: int
    max_branches: int

class ResolvedReference(BaseModel):
    title: str
    page_ref: Optional[str] = None
    summary: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    comparison: Optional[ComparisonResult] = None
    traversal_info: Optional[TraversalInfo] = None
    resolved_references: Optional[List[ResolvedReference]] = None
    hallucination_warning: Optional[Dict[str, Any]] = None

class TreeNode(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    page_ref: Optional[str] = None
    children: Optional[List['TreeNode']] = None

class TreeResponse(BaseModel):
    document_name: str
    tree: Dict[str, Any]