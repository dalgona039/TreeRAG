from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/tasks", tags=["tasks"])

CELERY_AVAILABLE = False
try:
    from src.tasks.indexing_tasks import (
        index_pdf,
        batch_index,
        get_task_status,
        revoke_task
    )
    CELERY_AVAILABLE = True
except ImportError:
    pass


class AsyncIndexRequest(BaseModel):
    filename: str


class BatchIndexRequest(BaseModel):
    filenames: List[str]


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None


@router.post("/index", response_model=TaskResponse)
async def async_index_pdf(request: AsyncIndexRequest) -> TaskResponse:
    if not CELERY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue not available. Use /api/index for synchronous indexing."
        )
    
    if not request.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must end with .pdf"
        )
    
    task = index_pdf.delay(request.filename)
    
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Indexing task queued for {request.filename}"
    )


@router.post("/index/batch", response_model=TaskResponse)
async def async_batch_index(request: BatchIndexRequest) -> TaskResponse:
    if not CELERY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue not available"
        )
    
    if not request.filenames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filenames provided"
        )
    
    invalid = [f for f in request.filenames if not f.lower().endswith('.pdf')]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filenames (must be .pdf): {invalid}"
        )
    
    task = batch_index.delay(request.filenames)
    
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Batch indexing queued for {len(request.filenames)} files"
    )


@router.get("/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    if not CELERY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue not available"
        )
    
    return get_task_status(task_id)


@router.delete("/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    if not CELERY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue not available"
        )
    
    success = revoke_task(task_id, terminate=True)
    
    if success:
        return {"task_id": task_id, "status": "cancelled"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )


@router.get("/")
async def list_active_tasks() -> Dict[str, Any]:
    if not CELERY_AVAILABLE:
        return {
            "available": False,
            "message": "Task queue not available"
        }
    
    try:
        from src.celery_app import celery_app
        
        inspect = celery_app.control.inspect()
        
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}
        
        total_active = sum(len(tasks) for tasks in active.values())
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())
        
        return {
            "available": True,
            "active_count": total_active,
            "reserved_count": total_reserved,
            "scheduled_count": total_scheduled,
            "workers": list(active.keys())
        }
    except Exception as e:
        return {
            "available": True,
            "error": str(e),
            "message": "Could not connect to workers"
        }
