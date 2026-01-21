from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IndexRequest(BaseModel):
    filename: str

class ChatRequest(BaseModel):
    question: str
    index_filenames: Optional[List[str]] = None
    enable_comparison: bool = True

class ComparisonResult(BaseModel):
    has_comparison: bool
    documents_compared: List[str]
    commonalities: Optional[str] = None
    differences: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    comparison: Optional[ComparisonResult] = None