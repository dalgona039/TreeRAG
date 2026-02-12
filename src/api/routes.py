import os
import shutil
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.config import Config
from src.core.indexer import RegulatoryIndexer
from src.core.reasoner import TreeRAGReasoner
from src.api.models import ChatRequest, ChatResponse, IndexRequest, ComparisonResult, TreeResponse, TraversalInfo
from src.utils.cache import get_cache
from src.utils.file_validator import validate_uploaded_file
from src.utils.hallucination_detector import HallucinationDetector

router = APIRouter()

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

def _route_documents(question: str, available_indices: List[str]) -> List[str]:
    if not available_indices:
        raise ValueError("No indexed documents available")
    
    if len(available_indices) == 1:
        return available_indices
    
    doc_summaries = []
    for filename in available_indices:
        filepath = os.path.join(Config.INDEX_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = json.load(f)
                doc_name = filename.replace("_index.json", "")
                summary = tree.get("summary", tree.get("title", doc_name))
                doc_summaries.append(f"- {doc_name}: {summary[:200]}")
        except:
            doc_summaries.append(f"- {filename.replace('_index.json', '')}")
    
    context = "\n".join(doc_summaries)
    
    prompt = f"""
당신은 문서 라우터입니다.
사용자의 질문을 분석하여, 어떤 규제 문서를 참조해야 하는지 선택하세요.

### 사용 가능한 문서:
{context}

### 사용자 질문:
{question}

### 규칙:
1. 질문과 가장 관련 있는 문서를 선택하세요.
2. 여러 문서가 관련되어 있다면 모두 선택하세요.
3. 반드시 위 목록에 있는 문서명만 사용하세요.
4. 응답 형식: JSON 배열로만 답하세요. 설명 없이 문서명만.

예시: ["2025학년도_교육과정_전자공학과", "교육과정_가이드라인"]

### 선택된 문서 (JSON 배열):
"""
    
    try:
        response = Config.CLIENT.models.generate_content(
            model=Config.MODEL_NAME,
            contents=prompt
        )
        
        result_text = response.text.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        selected_names = json.loads(result_text)
        
        if not isinstance(selected_names, list):
            selected_names = [selected_names]
        
        selected_files = []
        for name in selected_names:
            clean_name = name.strip()
            for filename in available_indices:
                doc_name = filename.replace("_index.json", "")
                if doc_name == clean_name or filename == clean_name:
                    selected_files.append(filename)
                    break
                if clean_name in doc_name or clean_name in filename:
                    selected_files.append(filename)
                    break
        
        if not selected_files:
            print(f"⚠️ Router couldn't match documents, using all")
            return available_indices
        
        print(f"📍 Router selected {len(selected_files)}/{len(available_indices)} documents: {selected_files}")
        return selected_files
        
    except Exception as e:
        print(f"⚠️ Router failed: {e}, using all documents")
        return available_indices

@router.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "TreeRAG API"}

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.pdf'}
ALLOWED_MIME_TYPES = {'application/pdf'}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    print(f"[DEBUG] Upload request - filename: {file.filename}")
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename != file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal detected"
        )
    
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are allowed. Got: {file_ext}"
        )
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type. Expected application/pdf, got {file.content_type}"
        )
    
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )
        
        is_valid, error_msg, validated_filename = validate_uploaded_file(
            contents, safe_filename, MAX_FILE_SIZE
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {error_msg}"
            )
        
        unique_filename = f"{uuid.uuid4().hex[:8]}_{validated_filename}"
        
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        file_path = os.path.join(Config.RAW_DATA_DIR, unique_filename)
        
        abs_file_path = os.path.abspath(file_path)
        abs_data_dir = os.path.abspath(Config.RAW_DATA_DIR)
        if not abs_file_path.startswith(abs_data_dir):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path"
            )
        
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        print(f"[DEBUG] File saved successfully: {unique_filename}")
        return {
            "message": "File uploaded successfully",
            "filename": unique_filename,
            "original_filename": safe_filename,
            "path": file_path,
            "size_bytes": len(contents)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed due to server error"
        )

@router.post("/index")
@limiter.limit("10/minute")
async def create_index(request: Request, req: IndexRequest) -> Dict[str, str]:
    print(f"[DEBUG] Index request for filename: {req.filename}")
    
    safe_filename = Path(req.filename).name
    if not safe_filename or safe_filename != req.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal detected"
        )
    
    if not safe_filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must end with .pdf"
        )
    
    pdf_path = os.path.join(Config.RAW_DATA_DIR, safe_filename)
    
    abs_pdf_path = os.path.abspath(pdf_path)
    abs_data_dir = os.path.abspath(Config.RAW_DATA_DIR)
    if not abs_pdf_path.startswith(abs_data_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    print(f"[DEBUG] Looking for PDF at: {pdf_path}")
    print(f"[DEBUG] File exists: {os.path.exists(pdf_path)}")
    
    if not os.path.exists(pdf_path):
        available_files = os.listdir(Config.RAW_DATA_DIR)
        print(f"[DEBUG] Available files in RAW_DATA_DIR: {available_files}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {safe_filename}"
        )
    
    index_filename = safe_filename.replace(".pdf", "_index.json").replace(".PDF", "_index.json")
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
        
        doc_title = safe_filename.replace(".pdf", "").replace(".PDF", "").replace("_", " ")
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
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    if req.index_filenames and len(req.index_filenames) > 0:
        selected_indices = req.index_filenames
        print(f"📌 Using user-specified documents: {selected_indices}")
    else:
        available_indices = [
            f for f in os.listdir(Config.INDEX_DIR)
            if f.endswith("_index.json") and os.path.isfile(os.path.join(Config.INDEX_DIR, f))
        ]
        
        if not available_indices:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No indexed documents found. Please upload and index documents first."
            )
        
        selected_indices = _route_documents(req.question, available_indices)
    
    for index_filename in selected_indices:
        if not index_filename.endswith("_index.json"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid index filename format: {index_filename}"
            )
    
    try:
        use_traversal = req.use_deep_traversal if req.use_deep_traversal is not None else Config.USE_DEEP_TRAVERSAL
        max_depth = req.max_depth if req.max_depth is not None else Config.MAX_TRAVERSAL_DEPTH
        max_branches = req.max_branches if req.max_branches is not None else Config.MAX_BRANCHES_PER_LEVEL
        domain_template = req.domain_template if req.domain_template else "general"
        language = req.language if req.language else "ko"
        
        reasoner = TreeRAGReasoner(
            selected_indices,
            use_deep_traversal=use_traversal
        )
        
        if req.node_context:
            enhanced_question = f"""[컨텍스트: 문서 섹션 "{req.node_context.get('title', '')}"]

사용자가 위 섹션에 대해 질문하고 있습니다.{f" (페이지: {req.node_context.get('page_ref', '')})" if req.node_context.get('page_ref') else ""}

질문: {req.question}

이 섹션과 관련된 내용을 중심으로 상세히 답변해주세요."""
            answer, traversal_info = reasoner.query(
                enhanced_question, 
                enable_comparison=req.enable_comparison,
                max_depth=max_depth,
                max_branches=max_branches,
                domain_template=domain_template,
                language=language
            )
        else:
            answer, traversal_info = reasoner.query(
                req.question, 
                enable_comparison=req.enable_comparison,
                max_depth=max_depth,
                max_branches=max_branches,
                domain_template=domain_template,
                language=language
            )
        
        citations = _extract_citations(answer)
        
        comparison = None
        if len(selected_indices) > 1 and req.enable_comparison:
            comparison = _extract_comparison(answer, selected_indices)
        
        trav_info = TraversalInfo(
            used_deep_traversal=traversal_info["used_deep_traversal"],
            nodes_visited=traversal_info["nodes_visited"],
            nodes_selected=traversal_info["nodes_selected"],
            max_depth=traversal_info["max_depth"],
            max_branches=traversal_info["max_branches"]
        )
        
        resolved_refs = None
        if "resolved_references" in traversal_info:
            from src.api.models import ResolvedReference
            resolved_refs = [
                ResolvedReference(**ref) 
                for ref in traversal_info["resolved_references"]
            ]
        
        hallucination_warning = None
        detector = HallucinationDetector(confidence_threshold=0.3)
        detection_result = detector.detect(answer, traversal_info["nodes_selected"])
        
        if not detection_result["is_reliable"] and len(detection_result["sentence_analysis"]) > 0:
            low_conf_count = sum(1 for s in detection_result["sentence_analysis"] if not s["is_grounded"])
            total_count = len(detection_result["sentence_analysis"])
            low_conf_ratio = low_conf_count / total_count if total_count > 0 else 0
            
            if low_conf_ratio >= 0.7:
                hallucination_warning = {
                    "message": f"{low_conf_count}/{total_count} sentences have low confidence",
                    "overall_confidence": detection_result["overall_confidence"],
                    "threshold": detector.confidence_threshold
                }
                print(f"⚠️ Hallucination detected: {low_conf_count}/{total_count} sentences low confidence (overall: {detection_result['overall_confidence']:.2f})")
        
        return ChatResponse(
            answer=answer, 
            citations=citations, 
            comparison=comparison,
            traversal_info=trav_info,
            resolved_references=resolved_refs,
            hallucination_warning=hallucination_warning
        )
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

@router.get("/tree/{index_filename}", response_model=TreeResponse)
async def get_tree_structure(index_filename: str) -> TreeResponse:
    if not index_filename.endswith("_index.json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid index filename format"
        )
    
    index_path = os.path.join(Config.INDEX_DIR, index_filename)
    
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index file not found: {index_filename}"
        )
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        doc_name = index_filename.replace("_index.json", "")
        
        return TreeResponse(
            document_name=doc_name,
            tree=tree_data
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON in index file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load tree: {str(e)}"
        )

@router.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    from urllib.parse import quote, unquote
    import logging
    
    logger = logging.getLogger(__name__)
    
    decoded_filename = unquote(filename)
    logger.info(f"[PDF Request] Original: {filename}")
    logger.info(f"[PDF Request] Decoded: {decoded_filename}")
    
    pdf_path = os.path.join(Config.RAW_DATA_DIR, decoded_filename)
    logger.info(f"[PDF Request] Trying exact match: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.warning(f"[PDF Request] Exact match failed, searching for similar files...")
        
        try:
            all_files = os.listdir(Config.RAW_DATA_DIR)
            pdf_files = [f for f in all_files if f.endswith('.pdf')]
            
            logger.info(f"[PDF Request] Available PDF files: {pdf_files}")
            
            search_name = decoded_filename.replace('.pdf', '').lower()
            logger.info(f"[PDF Request] Searching for: {search_name}")
            
            for pdf_file in pdf_files:
                file_without_ext = pdf_file.replace('.pdf', '').lower()
                if (file_without_ext == search_name or 
                    search_name in file_without_ext or 
                    file_without_ext in search_name):
                    logger.info(f"[PDF Request] Found similar file: {pdf_file}")
                    pdf_path = os.path.join(Config.RAW_DATA_DIR, pdf_file)
                    decoded_filename = pdf_file
                    break
            
        except Exception as e:
            logger.error(f"[PDF Request] Error during file search: {str(e)}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"[PDF Request] File not found after all attempts: {decoded_filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {decoded_filename}"
        )
    
    logger.info(f"[PDF Request] Serving file: {pdf_path}")
    
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

def _extract_comparison(text: str, doc_names: List[str]) -> ComparisonResult:
    import re
    has_comparison = False
    commonalities = None
    differences = None
    
    commonalities_match = re.search(r'\*\*1\.\s*공통점.*?\*\*(.+?)(?=\*\*2\.|📚|$)', text, re.DOTALL | re.IGNORECASE)
    if commonalities_match:
        commonalities = commonalities_match.group(1).strip()
        has_comparison = True
    
    differences_match = re.search(r'\*\*2\.\s*차이점.*?\*\*(.+?)(?=\*\*3\.|📚|$)', text, re.DOTALL | re.IGNORECASE)
    if differences_match:
        differences = differences_match.group(1).strip()
        has_comparison = True
    
    if not has_comparison:
        table_match = re.search(r'\|.*?\|.*?\|.*?\n\|[-:]+\|[-:]+\|[-:]+\|', text)
        if table_match:
            has_comparison = True
            differences = text[table_match.start():]
    
    clean_doc_names = [d.replace("_index.json", "") for d in doc_names]
    
    return ComparisonResult(
        has_comparison=has_comparison,
        documents_compared=clean_doc_names,
        commonalities=commonalities,
        differences=differences
    )

@router.get("/cache/stats")
async def get_cache_stats():
    cache = get_cache()
    stats = cache.get_stats()
    return {
        "status": "success",
        "cache_stats": stats
    }


@router.post("/cache/clear")
async def clear_cache():
    cache = get_cache()
    cache.clear()
    return {
        "status": "success",
        "message": "Cache cleared successfully"
    }


@router.post("/graph/build/{document_name}")
async def build_reasoning_graph(
    document_name: str,
    infer_edges: bool = True,
    max_edge_distance: int = 2
):
    from src.core.reasoning_graph import ReasoningGraph
    
    index_file = f"{document_name}_index.json"
    index_path = os.path.join(Config.INDEX_DIR, index_file)
    
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found in indices"
        )
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            tree = json.load(f)
        
        graph = ReasoningGraph(document_name)
        graph.build_from_tree(tree, infer_edges=infer_edges, max_edge_distance=max_edge_distance)
        
        graph_path = os.path.join(Config.INDEX_DIR, f"{document_name}_graph.json")
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "document_name": document_name,
            "graph_stats": graph.to_dict()["stats"],
            "graph_path": graph_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build reasoning graph: {str(e)}"
        )


@router.get("/graph/{document_name}")
async def get_reasoning_graph(document_name: str):
    from src.core.reasoning_graph import ReasoningGraph
    
    graph_path = os.path.join(Config.INDEX_DIR, f"{document_name}_graph.json")
    
    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404,
            detail=f"Reasoning graph for '{document_name}' not found. "
                   f"Build it first using POST /graph/build/{document_name}"
        )
    
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        return {
            "status": "success",
            "graph": graph_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load reasoning graph: {str(e)}"
        )


@router.post("/graph/{document_name}/search")
async def search_with_reasoning(
    document_name: str,
    query: str,
    max_hops: int = 3,
    top_k: int = 5
):
    from src.core.reasoning_graph import ReasoningGraph, GraphNavigator
    
    graph_path = os.path.join(Config.INDEX_DIR, f"{document_name}_graph.json")
    
    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404,
            detail=f"Reasoning graph for '{document_name}' not found"
        )
    
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        graph = ReasoningGraph.from_dict(graph_data)
        navigator = GraphNavigator(graph)
        
        result = navigator.search_with_reasoning(
            query=query,
            max_hops=max_hops,
            top_k=top_k
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning search failed: {str(e)}"
        )


@router.get("/graph/{document_name}/explain")
async def explain_connection(
    document_name: str,
    node_a: str,
    node_b: str
):
    from src.core.reasoning_graph import ReasoningGraph, GraphNavigator
    
    graph_path = os.path.join(Config.INDEX_DIR, f"{document_name}_graph.json")
    
    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404,
            detail=f"Reasoning graph for '{document_name}' not found"
        )
    
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        graph = ReasoningGraph.from_dict(graph_data)
        navigator = GraphNavigator(graph)
        
        explanation = navigator.explain_connection(node_a, node_b)
        
        if explanation is None:
            raise HTTPException(
                status_code=404,
                detail=f"One or both nodes not found: {node_a}, {node_b}"
            )
        
        return {
            "status": "success",
            "explanation": explanation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain connection: {str(e)}"
        )


@router.get("/graph/{document_name}/node/{node_id}")
async def get_node_context(
    document_name: str,
    node_id: str,
    include_neighbors: bool = True
):
    from src.core.reasoning_graph import ReasoningGraph
    
    graph_path = os.path.join(Config.INDEX_DIR, f"{document_name}_graph.json")
    
    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404,
            detail=f"Reasoning graph for '{document_name}' not found"
        )
    
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        graph = ReasoningGraph.from_dict(graph_data)
        context = graph.get_node_context(node_id, include_neighbors)
        
        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"Node '{node_id}' not found in graph"
            )
        
        return {
            "status": "success",
            "context": context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get node context: {str(e)}"
        )


# ==================== Benchmark Endpoints ====================

@router.get("/benchmark/domains")
async def list_available_domains():
    """List all available benchmark domains."""
    from src.core.domain_benchmark import DocumentDomain
    
    return {
        "status": "success",
        "domains": [d.value for d in DocumentDomain]
    }


@router.post("/benchmark/{document_name}/classify")
async def classify_document_domain(document_name: str):
    """
    Classify the domain of an indexed document.
    
    Analyzes document content and determines the most likely domain
    (medical, legal, technical, etc.).
    """
    from src.core.domain_benchmark import DomainClassifier
    
    index_path = os.path.join(Config.INDEX_DIR, f"{document_name}_index.json")
    
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found"
        )
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            tree = json.load(f)
        
        # Extract text for classification
        title = tree.get("title", document_name)
        summary = tree.get("summary", "")
        
        # Collect child summaries for more context
        children_text = []
        for child in tree.get("children", [])[:10]:
            children_text.append(child.get("summary", ""))
        
        full_text = f"{summary} {' '.join(children_text)}"
        
        # Classify using LLM for better accuracy
        domain, confidence = DomainClassifier.classify_with_llm(full_text, title)
        
        # Also get keyword-based classification for comparison
        keyword_domain, keyword_conf = DomainClassifier.classify(full_text, title)
        
        return {
            "status": "success",
            "document_name": document_name,
            "classification": {
                "domain": domain.value,
                "confidence": round(confidence, 4),
                "method": "llm"
            },
            "keyword_classification": {
                "domain": keyword_domain.value,
                "confidence": round(keyword_conf, 4),
                "method": "keyword"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/benchmark/{document_name}/run")
async def run_benchmark(
    document_name: str,
    domain: Optional[str] = None,
    use_reasoning: bool = False
):
    """
    Run benchmark evaluation for a document.
    
    If domain is not specified, it will be auto-detected.
    Generates benchmark questions and evaluates TreeRAG performance.
    """
    from src.core.domain_benchmark import (
        DocumentDomain, DomainClassifier, DomainBenchmark
    )
    
    index_path = os.path.join(Config.INDEX_DIR, f"{document_name}_index.json")
    
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found"
        )
    
    try:
        # Determine domain
        if domain:
            doc_domain = DocumentDomain.from_string(domain)
        else:
            with open(index_path, 'r', encoding='utf-8') as f:
                tree = json.load(f)
            title = tree.get("title", document_name)
            summary = tree.get("summary", "")
            doc_domain, _ = DomainClassifier.classify(summary, title)
        
        # Run benchmark
        benchmark = DomainBenchmark()
        report = benchmark.run_benchmark(
            document_name=document_name,
            domain=doc_domain,
            use_reasoning=use_reasoning
        )
        
        # Save report
        report_path = benchmark.save_report(report)
        
        return {
            "status": "success",
            "report": report.to_dict(),
            "report_path": report_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


@router.get("/benchmark/{document_name}/compare")
async def compare_domain_performance(document_name: str):
    """
    Compare benchmark performance across domains for a document.
    
    Returns rankings by accuracy, response time, and hallucination rate.
    """
    from src.core.domain_benchmark import DomainBenchmark
    
    benchmark = DomainBenchmark()
    historical = benchmark.load_historical_reports(document_name=document_name)
    
    if not historical:
        raise HTTPException(
            status_code=404,
            detail=f"No benchmark reports found for '{document_name}'"
        )
    
    # Load reports into benchmark results
    for report in historical:
        benchmark.results[document_name].append(report)
    
    comparison = benchmark.compare_domains(document_name)
    
    return {
        "status": "success",
        "comparison": comparison
    }


@router.get("/benchmark/reports")
async def list_benchmark_reports(
    document_name: Optional[str] = None,
    domain: Optional[str] = None
):
    """List all available benchmark reports."""
    from src.core.domain_benchmark import DomainBenchmark, DocumentDomain
    
    benchmark = DomainBenchmark()
    
    doc_domain = DocumentDomain.from_string(domain) if domain else None
    reports = benchmark.load_historical_reports(
        document_name=document_name,
        domain=doc_domain
    )
    
    return {
        "status": "success",
        "report_count": len(reports),
        "reports": [
            {
                "document_name": r.document_name,
                "domain": r.domain.value,
                "accuracy": r.accuracy,
                "total_questions": r.total_questions,
                "run_timestamp": r.run_timestamp
            }
            for r in reports
        ]
    }


@router.post("/benchmark/dataset/{domain}/add")
async def add_benchmark_question(
    domain: str,
    question: str,
    expected_answer: str,
    difficulty: str = "medium"
):
    """Add a new benchmark question to a domain dataset."""
    from src.core.domain_benchmark import BenchmarkDataset, DocumentDomain
    
    doc_domain = DocumentDomain.from_string(domain)
    
    if doc_domain == DocumentDomain.GENERAL and domain != "general":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain: {domain}"
        )
    
    try:
        dataset = BenchmarkDataset()
        new_question = dataset.add_question(
            domain=doc_domain,
            question=question,
            expected_answer=expected_answer,
            difficulty=difficulty
        )
        
        # Save the updated dataset
        all_questions = dataset.questions[domain]
        dataset.save_dataset(doc_domain, all_questions)
        
        return {
            "status": "success",
            "question_id": new_question.id,
            "message": f"Question added to {domain} benchmark dataset"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add question: {str(e)}"
        )


@router.get("/benchmark/dataset/{domain}")
async def get_benchmark_dataset(domain: str):
    """Get all benchmark questions for a domain."""
    from src.core.domain_benchmark import BenchmarkDataset, DocumentDomain
    
    doc_domain = DocumentDomain.from_string(domain)
    
    dataset = BenchmarkDataset()
    questions = dataset.load_dataset(doc_domain)
    
    return {
        "status": "success",
        "domain": domain,
        "question_count": len(questions),
        "questions": [q.to_dict() for q in questions]
    }

