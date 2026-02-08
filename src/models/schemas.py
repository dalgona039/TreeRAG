"""Pydantic schemas for data validation."""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class PageNode(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., min_length=1, description="Node title")
    summary: str = Field(default="", description="Node content summary")
    page_ref: str = Field(default="", description="Page reference (e.g., '12-15')")
    text: Optional[str] = Field(default=None, description="Full text content")
    children: Optional[List['PageNode']] = Field(default=None, description="Child nodes")
    
    @validator('children', pre=True, always=True)
    def validate_children(cls, v):
        if v is None or v == []:
            return None
        return v
    
    class Config:
        extra = 'allow'


PageNode.model_rebuild()
