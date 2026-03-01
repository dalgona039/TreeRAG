import json
import os
from threading import Lock
from typing import Any, Dict, List, Optional


class SessionRepository:
    def __init__(self, storage_path: str = "data/sessions/sessions.json"):
        self.storage_path = storage_path
        self._lock = Lock()
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if not os.path.exists(self.storage_path):
                return {"sessions": [], "currentSessionId": None}

            try:
                with open(self.storage_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                return {"sessions": [], "currentSessionId": None}

            sessions = data.get("sessions", [])
            current_session_id = data.get("currentSessionId")

            if not isinstance(sessions, list):
                sessions = []

            if current_session_id is not None and not isinstance(current_session_id, str):
                current_session_id = None

            return {
                "sessions": sessions,
                "currentSessionId": current_session_id,
            }

    def save(self, sessions: List[Dict[str, Any]], current_session_id: Optional[str]) -> Dict[str, Any]:
        payload = {
            "sessions": sessions if isinstance(sessions, list) else [],
            "currentSessionId": current_session_id if isinstance(current_session_id, str) else None,
        }

        with self._lock:
            temp_path = f"{self.storage_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
            os.replace(temp_path, self.storage_path)

        return payload
