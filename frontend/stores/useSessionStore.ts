import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatSession, Message } from "@/lib/types";

const normalizeSession = (raw: Partial<ChatSession>): ChatSession => {
  const createdAt = raw.createdAt instanceof Date
    ? raw.createdAt
    : new Date(raw.createdAt ?? Date.now());

  return {
    id: String(raw.id ?? Date.now()),
    title: typeof raw.title === "string" && raw.title.trim() ? raw.title : "Untitled Session",
    indexFiles: Array.isArray(raw.indexFiles) ? raw.indexFiles : [],
    originalFilenames: Array.isArray(raw.originalFilenames) ? raw.originalFilenames : undefined,
    messages: Array.isArray(raw.messages) ? raw.messages : [],
    createdAt: Number.isNaN(createdAt.getTime()) ? new Date() : createdAt,
  };
};

interface SessionState {
  // Session data
  sessions: ChatSession[];
  currentSessionId: string | null;
  
  // Computed
  currentSession: ChatSession | undefined;
  
  // Actions
  setSessions: (sessions: ChatSession[] | ((prev: ChatSession[]) => ChatSession[])) => void;
  setCurrentSessionId: (id: string | null) => void;
  createNewSession: () => void;
  deleteSession: (sessionId: string) => void;
  addMessageToSession: (sessionId: string, message: Message) => void;
  updateSession: (sessionId: string, updates: Partial<ChatSession>) => void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,
      
      get currentSession() {
        const state = get();
        return state.sessions.find(s => s.id === state.currentSessionId);
      },
      
      setSessions: (sessionsOrUpdater) => {
        set((state) => {
          const newSessions = typeof sessionsOrUpdater === 'function' 
            ? sessionsOrUpdater(state.sessions)
            : sessionsOrUpdater;
          return { sessions: newSessions };
        });
      },
      
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
      
      createNewSession: () => set({ currentSessionId: null }),
      
      deleteSession: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.filter(s => s.id !== sessionId),
          currentSessionId: state.currentSessionId === sessionId ? null : state.currentSessionId,
        }));
      },
      
      addMessageToSession: (sessionId, message) => {
        set((state) => ({
          sessions: state.sessions.map(s => 
            s.id === sessionId 
              ? { ...s, messages: [...s.messages, message] }
              : s
          ),
        }));
      },
      
      updateSession: (sessionId, updates) => {
        set((state) => ({
          sessions: state.sessions.map(s => 
            s.id === sessionId ? { ...s, ...updates } : s
          ),
        }));
      },
    }),
    {
      name: "treerag-sessions",
      partialize: (state) => ({ 
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
      }),
      merge: (persistedState, currentState) => {
        const persisted = (persistedState ?? {}) as Partial<SessionState>;
        const normalizedSessions = Array.isArray(persisted.sessions)
          ? persisted.sessions.map((session) => normalizeSession(session as Partial<ChatSession>))
          : currentState.sessions;

        const preferredSessionId = persisted.currentSessionId ?? null;
        const hasPreferredSession = preferredSessionId
          ? normalizedSessions.some((session) => session.id === preferredSessionId)
          : false;

        return {
          ...currentState,
          ...persisted,
          sessions: normalizedSessions,
          currentSessionId: hasPreferredSession
            ? preferredSessionId
            : (normalizedSessions[0]?.id ?? null),
        };
      },
    }
  )
);
