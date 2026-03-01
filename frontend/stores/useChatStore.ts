import { create } from "zustand";
import { toast } from "react-hot-toast";
import { api } from "@/lib/api";
import { useSessionStore } from "./useSessionStore";
import { useSettingsStore } from "./useSettingsStore";
import { useUIStore } from "./useUIStore";
import { usePerformanceStore } from "./usePerformanceStore";
import type { Message } from "@/lib/types";

const estimateContextTokens = (traversalInfo: any): number => {
  if (!traversalInfo) return 0;
  if (typeof traversalInfo.context_tokens === "number") return traversalInfo.context_tokens;
  if (typeof traversalInfo.total_tokens === "number") return traversalInfo.total_tokens;

  const selectedNodes = Array.isArray(traversalInfo.nodes_selected) ? traversalInfo.nodes_selected : [];
  const totalChars = selectedNodes.reduce((acc: number, node: any) => {
    const title = typeof node?.title === "string" ? node.title : "";
    const content = typeof node?.content === "string" ? node.content : "";
    return acc + title.length + content.length;
  }, 0);

  return Math.floor(totalChars / 4);
};

interface ChatState {
  // Input state
  input: string;
  isGenerating: boolean;
  
  // Actions
  setInput: (input: string) => void;
  sendMessage: () => Promise<void>;
  clearInput: () => void;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  input: "",
  isGenerating: false,
  
  setInput: (input) => set({ input }),
  clearInput: () => set({ input: "" }),
  
  sendMessage: async () => {
    const { input, isGenerating } = get();
    if (!input.trim() || isGenerating) return;
    
    // Get state from other stores
    const sessionStore = useSessionStore.getState();
    const settingsStore = useSettingsStore.getState();
    const uiStore = useUIStore.getState();
    const performanceStore = usePerformanceStore.getState();
    
    const { currentSessionId, sessions, setSessions } = sessionStore;
    const currentSession = sessions.find(s => s.id === currentSessionId);
    
    if (!currentSessionId || !currentSession) {
      toast.error("No active session");
      return;
    }
    
    const userMsg = input;
    set({ input: "", isGenerating: true });
    
    // Add user message
    const updatedMessages: Message[] = [
      ...currentSession.messages,
      { role: "user", content: userMsg }
    ];
    
    setSessions(prev => prev.map(session => 
      session.id === currentSessionId 
        ? { ...session, messages: updatedMessages }
        : session
    ));
    
    const startTime = Date.now();
    
    try {
      const requestBody = {
        question: userMsg,
        index_filenames: currentSession.indexFiles,
        use_deep_traversal: settingsStore.useDeepTraversal,
        max_depth: settingsStore.maxDepth,
        max_branches: settingsStore.maxBranches,
        domain_template: settingsStore.domainTemplate,
        language: settingsStore.language,
        node_context: uiStore.selectedNode ? {
          id: uiStore.selectedNode.id,
          title: uiStore.selectedNode.title,
          page_ref: uiStore.selectedNode.page_ref,
          summary: uiStore.selectedNode.summary,
        } : undefined,
      };
      
      const res = await api.chat(requestBody);
      
      const botMsg = res.data.answer;
      const citations = res.data.citations || [];
      const comparison = res.data.comparison || null;
      const traversal_info = res.data.traversal_info || null;
      const resolved_references = res.data.resolved_references || null;
      const hallucination_warning = res.data.hallucination_warning || null;
      
      // Record performance metrics
      const responseTime = (Date.now() - startTime) / 1000;
      const contextSize = estimateContextTokens(traversal_info);
      
      performanceStore.recordQuery({
        responseTime,
        contextSize,
        useDeepTraversal: settingsStore.useDeepTraversal,
      });
      
      // Update session with bot message
      setSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? {
              ...session,
              messages: [
                ...session.messages,
                {
                  role: "assistant" as const,
                  content: botMsg,
                  citations,
                  comparison,
                  traversal_info,
                  resolved_references,
                  hallucination_warning,
                }
              ]
            }
          : session
      ));
      
    } catch (error: unknown) {
      console.error("Chat error:", error);
      
      const errorMessage = error instanceof Error 
        ? error.message 
        : "An error occurred";
      
      toast.error(errorMessage);
      
      // Restore original messages on error
      setSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { ...session, messages: currentSession.messages }
          : session
      ));
    } finally {
      set({ isGenerating: false });
    }
  },
}));
