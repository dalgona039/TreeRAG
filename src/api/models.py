from pydantic import BaseModel
from typing import List, Optional

class IndexRequest(BaseModel):
    filename: str

class ChatRequest(BaseModel):
    question: str
    index_filenames: List[str]

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]