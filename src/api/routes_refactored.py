import os
import json
from typing import Dict, List, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from urllib.parse import unquote
import logging

from src.config import Config
from src.api.models import (
    ChatRequest, ChatResponse, IndexRequest, 
    ComparisonResult, TreeResponse, TraversalInfo, ResolvedReference,
    SessionSyncRequest, SessionSyncResponse
)
from src.services import UploadService, IndexService, ChatService
from src.services.chat_service import NodeContext
from src.repositories import DocumentRepository, IndexRepository, SessionRepository
from src.utils.cache import get_cache


router = APIRouter()
logger = logging.getLogger(__name__)

def get_real_ip(request: Request) -> str:
    TRUSTED_PROXIES = {'127.0.0.1', 'localhost'}
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for and request.client and request.client.host in TRUSTED_PROXIES:
        first_ip = forwarded_for.split(",")[0].strip()
        return first_ip
    
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=get_real_ip)


document_repo = DocumentRepository()
index_repo = IndexRepository()
session_repo = SessionRepository()

upload_service = UploadService(document_repo)
index_service = IndexService(document_repo, index_repo)
chat_service = ChatService(index_repo)


@router.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "TreeRAG API"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    logger.info(f"[Upload] Request - filename: {file.filename}")
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    try:
        contents = await file.read()
        
        result = upload_service.upload_file(
            content=contents,
            filename=file.filename,
            content_type=file.content_type
        )
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        logger.info(f"[Upload] Success: {result.filename}")
        return {
            "message": "File uploaded successfully",
            "filename": result.filename,
            "original_filename": result.original_filename,
            "path": result.file_path,
            "size_bytes": result.size_bytes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed due to server error"
        )


@router.post("/index")
@limiter.limit("10/minute")
async def create_index(request: Request, req: IndexRequest) -> Dict[str, str]:
    logger.info(f"[Index] Request for: {req.filename}")
    
    result = index_service.create_index(req.filename)
    
    if not result.success:
        if "not found" in (result.error_message or "").lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error_message
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error_message
        )
    
    logger.info(f"[Index] {result.status}: {result.index_filename}")
    return {
        "message": f"Indexing {'completed' if result.status == 'created' else 'already exists'}",
        "index_file": result.index_filename or "",
        "status": result.status
    }


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    node_context = None
    if req.node_context:
        node_context = NodeContext(
            id=req.node_context.get('id', ''),
            title=req.node_context.get('title', ''),
            page_ref=req.node_context.get('page_ref'),
            summary=req.node_context.get('summary')
        )
    
    result = chat_service.chat(
        question=req.question,
        index_filenames=req.index_filenames,
        use_deep_traversal=req.use_deep_traversal,
        max_depth=req.max_depth,
        max_branches=req.max_branches,
        domain_template=req.domain_template or "general",
        language=req.language or "auto",
        node_context=node_context,
        enable_comparison=req.enable_comparison or False
    )
    
    if not result.success:
        if "not found" in (result.error_message or "").lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error_message
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error_message
        )
    
    comparison = None
    if result.comparison:
        comparison = ComparisonResult(
            has_comparison=result.comparison.has_comparison,
            documents_compared=result.comparison.documents_compared,
            commonalities=result.comparison.commonalities,
            differences=result.comparison.differences
        )
    
    traversal_info = None
    if result.traversal_info:
        traversal_info = TraversalInfo(
            used_deep_traversal=result.traversal_info.used_deep_traversal,
            nodes_visited=result.traversal_info.nodes_visited,
            nodes_selected=result.traversal_info.nodes_selected,
            max_depth=result.traversal_info.max_depth,
            max_branches=result.traversal_info.max_branches,
            context_tokens=result.traversal_info.context_tokens,
            total_tokens=result.traversal_info.total_tokens
        )
    
    resolved_refs = None
    if result.resolved_references:
        resolved_refs = [
            ResolvedReference(
                title=ref.title,
                page_ref=ref.page_ref,
                summary=ref.summary
            )
            for ref in result.resolved_references
        ]
    
    hallucination_warning = None
    if result.hallucination_warning:
        hallucination_warning = {
            "message": result.hallucination_warning.message,
            "overall_confidence": result.hallucination_warning.overall_confidence,
            "threshold": result.hallucination_warning.threshold
        }
    
    return ChatResponse(
        answer=result.answer or "",
        citations=result.citations,
        comparison=comparison,
        traversal_info=traversal_info,
        resolved_references=resolved_refs,
        hallucination_warning=hallucination_warning
    )


@router.get("/indices")
async def list_indices() -> Dict[str, List[str]]:
    indices = index_repo.list_all()
    return {"indices": sorted(indices)}


@router.get("/sessions", response_model=SessionSyncResponse)
async def load_sessions() -> SessionSyncResponse:
    data = session_repo.load()
    return SessionSyncResponse(
        sessions=data.get("sessions", []),
        currentSessionId=data.get("currentSessionId")
    )


@router.put("/sessions", response_model=SessionSyncResponse)
async def save_sessions(req: SessionSyncRequest) -> SessionSyncResponse:
    saved = session_repo.save(req.sessions, req.currentSessionId)
    return SessionSyncResponse(
        sessions=saved.get("sessions", []),
        currentSessionId=saved.get("currentSessionId")
    )

@router.get("/pdfs")
async def list_pdfs() -> Dict[str, List[str]]:
    pdfs = document_repo.list_all()
    return {"pdfs": sorted(pdfs)}


@router.get("/tree/{index_filename}", response_model=TreeResponse)
async def get_tree_structure(index_filename: str) -> TreeResponse:
    if not index_filename.endswith("_index.json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid index filename format"
        )
    
    tree_data = index_repo.load(index_filename)
    
    if not tree_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index file not found: {index_filename}"
        )
    
    doc_name = index_filename.replace("_index.json", "")
    return TreeResponse(document_name=doc_name, tree=tree_data)


@router.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    decoded_filename = unquote(filename)
    logger.info(f"[PDF] Request: {decoded_filename}")
    
    pdf_path = document_repo.get_path(decoded_filename)
    
    if not pdf_path:
        logger.warning(f"[PDF] Exact match failed, searching...")
        all_pdfs = document_repo.list_all()
        search_name = decoded_filename.replace('.pdf', '').lower()
        
        for pdf_file in all_pdfs:
            file_without_ext = pdf_file.replace('.pdf', '').lower()
            if (file_without_ext == search_name or 
                search_name in file_without_ext or 
                file_without_ext in search_name):
                pdf_path = document_repo.get_path(pdf_file)
                logger.info(f"[PDF] Found similar: {pdf_file}")
                break
    
    if not pdf_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {decoded_filename}"
        )
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.get("/cache/stats")
async def get_cache_stats():
    cache = get_cache()
    stats = cache.get_stats()
    return {"status": "success", "cache_stats": stats}

@router.post("/cache/clear")
async def clear_cache():
    cache = get_cache()
    cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}
