import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import type { ChatSession } from "@/lib/types";
import { STORAGE_KEY } from "@/lib/api";

export function useSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const currentSession = sessions.find(s => s.id === currentSessionId);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSessions(parsed.map((s: ChatSession) => ({
          ...s,
          createdAt: new Date(s.createdAt)
        })));
      } catch (error) {
        console.error("Failed to load sessions:", error);
      }
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }
  }, [sessions]);

  const createNewSession = () => {
    setCurrentSessionId(null);
  };

  const deleteSession = (sessionId: string, t: any) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
    }
    toast.success(t.sessionDeleted);
  };

  return {
    sessions,
    setSessions,
    currentSessionId,
    setCurrentSessionId,
    currentSession,
    createNewSession,
    deleteSession,
  };
}
