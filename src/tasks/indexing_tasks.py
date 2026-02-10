import time
from typing import Dict, Any, List, Optional
from celery import shared_task, current_task
from celery.result import AsyncResult

from src.celery_app import celery_app


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def index_pdf(self, pdf_filename: str) -> Dict[str, Any]:
    task_id = self.request.id
    
    try:
        self.update_state(state="PROGRESS", meta={
            "stage": "initializing",
            "progress": 0,
            "message": f"Starting indexing for {pdf_filename}"
        })
        
        from src.services import IndexService
        from src.repositories import DocumentRepository, IndexRepository
        
        doc_repo = DocumentRepository()
        index_repo = IndexRepository()
        index_service = IndexService(doc_repo, index_repo)
        
        self.update_state(state="PROGRESS", meta={
            "stage": "extracting",
            "progress": 20,
            "message": "Extracting text from PDF"
        })
        
        result = index_service.create_index(pdf_filename)
        
        if not result.success:
            return {
                "status": "failed",
                "task_id": task_id,
                "filename": pdf_filename,
                "error": result.error_message
            }
        
        self.update_state(state="PROGRESS", meta={
            "stage": "completed",
            "progress": 100,
            "message": "Indexing completed"
        })
        
        return {
            "status": "completed",
            "task_id": task_id,
            "filename": pdf_filename,
            "index_filename": result.index_filename,
            "index_status": result.status
        }
        
    except Exception as e:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {
            "status": "failed",
            "task_id": task_id,
            "filename": pdf_filename,
            "error": str(e)
        }


@shared_task(bind=True)
def batch_index(self, pdf_filenames: List[str]) -> Dict[str, Any]:
    task_id = self.request.id
    results = []
    total = len(pdf_filenames)
    
    for i, filename in enumerate(pdf_filenames):
        self.update_state(state="PROGRESS", meta={
            "stage": "processing",
            "current": i + 1,
            "total": total,
            "progress": int((i / total) * 100),
            "current_file": filename
        })
        
        try:
            from src.services import IndexService
            from src.repositories import DocumentRepository, IndexRepository
            
            doc_repo = DocumentRepository()
            index_repo = IndexRepository()
            index_service = IndexService(doc_repo, index_repo)
            
            result = index_service.create_index(filename)
            
            results.append({
                "filename": filename,
                "success": result.success,
                "index_filename": result.index_filename,
                "status": result.status,
                "error": result.error_message
            })
            
        except Exception as e:
            results.append({
                "filename": filename,
                "success": False,
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r.get("success"))
    
    return {
        "status": "completed",
        "task_id": task_id,
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "results": results
    }


def get_task_status(task_id: str) -> Dict[str, Any]:
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }
    
    if result.state == "PENDING":
        response["message"] = "Task is waiting to be processed"
    elif result.state == "PROGRESS":
        response["progress"] = result.info
    elif result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.result)
    elif result.state == "REVOKED":
        response["message"] = "Task was cancelled"
    
    return response


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        return True
    except Exception:
        return False
