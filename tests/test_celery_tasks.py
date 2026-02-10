import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.task_routes import router, CELERY_AVAILABLE


class TestTaskRoutesWithoutCelery:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.mark.skipif(CELERY_AVAILABLE, reason="Celery is available")
    def test_async_index_without_celery(self, client):
        response = client.post("/tasks/index", json={"filename": "test.pdf"})
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
    
    @pytest.mark.skipif(CELERY_AVAILABLE, reason="Celery is available")
    def test_batch_index_without_celery(self, client):
        response = client.post("/tasks/index/batch", json={"filenames": ["a.pdf", "b.pdf"]})
        assert response.status_code == 503
    
    @pytest.mark.skipif(CELERY_AVAILABLE, reason="Celery is available")
    def test_get_task_without_celery(self, client):
        response = client.get("/tasks/some-task-id")
        assert response.status_code == 503
    
    @pytest.mark.skipif(CELERY_AVAILABLE, reason="Celery is available")
    def test_list_tasks_without_celery(self, client):
        response = client.get("/tasks/")
        assert response.status_code == 200
        assert response.json()["available"] is False


class TestTaskRoutesValidation:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_invalid_filename_extension(self, client):
        with patch("src.api.task_routes.index_pdf") as mock_task:
            response = client.post("/tasks/index", json={"filename": "test.txt"})
            assert response.status_code == 400
            assert "must end with .pdf" in response.json()["detail"]
    
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_empty_batch_filenames(self, client):
        response = client.post("/tasks/index/batch", json={"filenames": []})
        assert response.status_code == 400
        assert "No filenames" in response.json()["detail"]
    
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_batch_invalid_extensions(self, client):
        response = client.post("/tasks/index/batch", json={
            "filenames": ["valid.pdf", "invalid.txt"]
        })
        assert response.status_code == 400
        assert "invalid.txt" in response.json()["detail"]


class TestCeleryAppConfig:
    def test_celery_app_import(self):
        try:
            from src.celery_app import celery_app, get_celery_app
            assert celery_app is not None
            assert get_celery_app() is celery_app
        except ImportError:
            pytest.skip("Celery not installed")
    
    def test_celery_config(self):
        try:
            from src.celery_app import celery_app
            
            assert celery_app.conf.task_serializer == "json"
            assert celery_app.conf.result_serializer == "json"
            assert celery_app.conf.task_track_started is True
            assert celery_app.conf.task_time_limit == 600
        except ImportError:
            pytest.skip("Celery not installed")


class TestIndexingTasksModule:
    def test_tasks_import(self):
        try:
            from src.tasks import index_pdf, batch_index, get_task_status
            assert index_pdf is not None
            assert batch_index is not None
            assert get_task_status is not None
        except ImportError:
            pytest.skip("Celery not installed")


class TestGetTaskStatus:
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_get_pending_task_status(self):
        from src.tasks.indexing_tasks import get_task_status
        
        with patch("src.tasks.indexing_tasks.AsyncResult") as mock_result:
            mock_result.return_value.state = "PENDING"
            mock_result.return_value.ready.return_value = False
            mock_result.return_value.successful.return_value = False
            
            status = get_task_status("test-task-id")
            
            assert status["task_id"] == "test-task-id"
            assert status["state"] == "PENDING"
            assert "waiting" in status["message"]
    
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_get_progress_task_status(self):
        from src.tasks.indexing_tasks import get_task_status
        
        with patch("src.tasks.indexing_tasks.AsyncResult") as mock_result:
            mock_result.return_value.state = "PROGRESS"
            mock_result.return_value.ready.return_value = False
            mock_result.return_value.info = {
                "stage": "extracting",
                "progress": 50
            }
            
            status = get_task_status("test-task-id")
            
            assert status["state"] == "PROGRESS"
            assert status["progress"]["stage"] == "extracting"
            assert status["progress"]["progress"] == 50
    
    @pytest.mark.skipif(not CELERY_AVAILABLE, reason="Celery not installed")
    def test_get_success_task_status(self):
        from src.tasks.indexing_tasks import get_task_status
        
        with patch("src.tasks.indexing_tasks.AsyncResult") as mock_result:
            mock_result.return_value.state = "SUCCESS"
            mock_result.return_value.ready.return_value = True
            mock_result.return_value.successful.return_value = True
            mock_result.return_value.result = {
                "status": "completed",
                "index_filename": "test_index.json"
            }
            
            status = get_task_status("test-task-id")
            
            assert status["state"] == "SUCCESS"
            assert status["ready"] is True
            assert status["result"]["status"] == "completed"
