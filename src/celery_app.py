import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "treerag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks.indexing_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    
    worker_prefetch_multiplier=1,
    worker_concurrency=2,
    
    result_expires=3600,
    
    task_routes={
        "src.tasks.indexing_tasks.index_pdf": {"queue": "indexing"},
        "src.tasks.indexing_tasks.batch_index": {"queue": "indexing"},
    },
    
    task_default_queue="default",
)


def get_celery_app() -> Celery:
    return celery_app
