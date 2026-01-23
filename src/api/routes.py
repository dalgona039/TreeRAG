import os
import shutil
import json
from typing import Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from src.config import Config
from src.core.indexer import RegulatoryIndexer
from src.core.reasoner import RegulatoryReasoner
from src.api.models import ChatRequest, ChatResponse, IndexRequest, ComparisonResult, TreeResponse
from src.api.openai_compat import ChatCompletionRequest, ChatCompletionResponse, openai_to_gemini_chat_completion

router = APIRouter()

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
        reasoner = RegulatoryReasoner(selected_indices)
        
        if req.node_context:
            enhanced_question = f"""[컨텍스트: 문서 섹션 "{req.node_context.get('title', '')}"]

사용자가 위 섹션에 대해 질문하고 있습니다.{f" (페이지: {req.node_context.get('page_ref', '')})" if req.node_context.get('page_ref') else ""}

질문: {req.question}

이 섹션과 관련된 내용을 중심으로 상세히 답변해주세요."""
            answer = reasoner.query(enhanced_question, enable_comparison=req.enable_comparison)
        else:
            answer = reasoner.query(req.question, enable_comparison=req.enable_comparison)
        
        citations = _extract_citations(answer)
        
        comparison = None
        if len(selected_indices) > 1 and req.enable_comparison:
            comparison = _extract_comparison(answer, selected_indices)
        
        return ChatResponse(answer=answer, citations=citations, comparison=comparison)
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

@router.post("/v1/chat/completions")
async def openai_compatible_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI 호환 Chat Completion API
    Agent Lightning의 APO가 Gemini API를 사용할 수 있도록 변환
    """
    return await openai_to_gemini_chat_completion(request)