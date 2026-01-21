import os
import shutil
from typing import Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from src.config import Config
from src.core.indexer import RegulatoryIndexer
from src.core.reasoner import RegulatoryReasoner
from src.api.models import ChatRequest, ChatResponse, IndexRequest

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "Medi-Reg Master API"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    print(f"[DEBUG] Upload request - filename: {file.filename}")
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    try:
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        file_path = os.path.join(Config.RAW_DATA_DIR, file.filename)
        print(f"[DEBUG] Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[DEBUG] File saved successfully: {file.filename}")
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@router.post("/index")
async def create_index(req: IndexRequest) -> Dict[str, str]:
    print(f"[DEBUG] Index request for filename: {req.filename}")
    
    if not req.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must end with .pdf"
        )
    
    pdf_path = os.path.join(Config.RAW_DATA_DIR, req.filename)
    print(f"[DEBUG] Looking for PDF at: {pdf_path}")
    print(f"[DEBUG] File exists: {os.path.exists(pdf_path)}")
    
    if not os.path.exists(pdf_path):
        available_files = os.listdir(Config.RAW_DATA_DIR)
        print(f"[DEBUG] Available files in RAW_DATA_DIR: {available_files}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {req.filename}. Available files: {available_files}"
        )
    
    index_filename = req.filename.replace(".pdf", "_index.json").replace(".PDF", "_index.json")
    index_path = os.path.join(Config.INDEX_DIR, index_filename)
    
    if os.path.exists(index_path):
        return {
            "message": "Index already exists",
            "index_file": index_filename,
            "status": "existing"
        }
    
    try:
        indexer = RegulatoryIndexer()
        text = indexer.extract_text(pdf_path)
        
        doc_title = req.filename.replace(".pdf", "").replace(".PDF", "").replace("_", " ")
        tree = indexer.create_index(doc_title, text)
        
        if not tree:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Indexing failed: empty result"
            )
        
        indexer.save_index(tree, index_filename)
        
        return {
            "message": "Indexing completed successfully",
            "index_file": index_filename,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    if not req.index_filenames or len(req.index_filenames) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one index filename required"
        )
    
    for index_filename in req.index_filenames:
        if not index_filename.endswith("_index.json"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid index filename format: {index_filename}"
            )
    
    try:
        reasoner = RegulatoryReasoner(req.index_filenames)
        answer = reasoner.query(req.question)
        
        citations = _extract_citations(answer)
        
        return ChatResponse(answer=answer, citations=citations)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index file not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.get("/indices")
async def list_indices() -> Dict[str, List[str]]:
    try:
        if not os.path.exists(Config.INDEX_DIR):
            os.makedirs(Config.INDEX_DIR, exist_ok=True)
            return {"indices": []}
        
        files = [
            f for f in os.listdir(Config.INDEX_DIR)
            if f.endswith("_index.json") and os.path.isfile(os.path.join(Config.INDEX_DIR, f))
        ]
        print(f"[DEBUG] Available indices: {files}")
        return {"indices": sorted(files)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indices: {str(e)}"
        )

@router.get("/pdfs")
async def list_pdfs() -> Dict[str, List[str]]:
    try:
        if not os.path.exists(Config.RAW_DATA_DIR):
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            return {"pdfs": []}
        
        files = [
            f for f in os.listdir(Config.RAW_DATA_DIR)
            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(Config.RAW_DATA_DIR, f))
        ]
        print(f"[DEBUG] Available PDFs: {files}")
        return {"pdfs": sorted(files)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list PDFs: {str(e)}"
        )

def _extract_citations(text: str) -> List[str]:
    import re
    citations = []
    
    patterns = [
        r'\[([^\]]+?),\s*p\.(\d+(?:-\d+)?(?:,\s*p\.\d+(?:-\d+)?)*)\]',
        r'\[([^\]]+?)\s*-?\s*(?:Sec|Section)\s*[\d.]+,?\s*(?:Pg?|Page)\.?\s*(\d+(?:-\d+)?)\]',
        r'\[([^\]]+?),\s*(?:페이지|쪽)\s*(\d+(?:-\d+)?)\]',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            doc = match.group(1).strip()
            page = match.group(2).strip()
            citations.append(f"{doc}, p.{page}")
    
    if not citations:
        fallback_pattern = r'\[([^\]]+)\]'
        fallback_matches = re.findall(fallback_pattern, text)
        citations = [m.strip() for m in fallback_matches if m.strip()]
    
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)
    
    return unique_citations