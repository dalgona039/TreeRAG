import { useState } from "react";
import { toast } from "react-hot-toast";
import { api } from "@/lib/api";
import type { ChatSession, Message, ApiError, TreeNode } from "@/lib/types";

export function useChat(
  sessions: ChatSession[],
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>,
  currentSessionId: string | null,
  currentSession: ChatSession | undefined,
  selectedNode: TreeNode | null,
  useDeepTraversal: boolean,
  maxDepth: number,
  maxBranches: number,
  domainTemplate: string,
  language: string,
  setPerformanceMetrics: React.Dispatch<React.SetStateAction<any>>
) {
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleSendMessage = async () => {
    if (!input.trim() || !currentSessionId || !currentSession) return;

    const userMsg = input;
    setInput("");

    const updatedMessages: Message[] = [
      ...currentSession.messages,
      { role: "user", content: userMsg }
    ];

    setSessions(prev => prev.map(session => 
      session.id === currentSessionId 
        ? { ...session, messages: updatedMessages }
        : session
    ));
    
    setIsGenerating(true);
    const startTime = Date.now();

    try {
      const requestBody: any = {
        question: userMsg,
        index_filenames: currentSession.indexFiles,
        use_deep_traversal: useDeepTraversal,
        max_depth: maxDepth,
        max_branches: maxBranches,
        domain_template: domainTemplate,
        language: language,
      };
      
      if (selectedNode) {
        requestBody.node_context = {
          id: selectedNode.id,
          title: selectedNode.title,
          page_ref: selectedNode.page_ref,
          summary: selectedNode.summary,
        };
      }
      
      const res = await api.chat(requestBody);
      
      const botMsg = res.data.answer;
      const citations = res.data.citations || [];
      const comparison = res.data.comparison || null;
      const traversalInfo = res.data.traversal_info || null;
      const resolvedReferences = res.data.resolved_references || null;
      const hallucinationWarning = res.data.hallucination_warning || null;

      const responseTime = (Date.now() - startTime) / 1000;
      const contextSize = traversalInfo?.total_tokens || 0;

      setPerformanceMetrics((prev: any) => {
        const newHistory = [
          ...prev.queriesHistory,
          {
            timestamp: new Date(),
            responseTime,
            contextSize,
            useDeepTraversal
          }
        ].slice(-50);

        const totalQueries = prev.totalQueries + 1;
        const avgResponseTime = (prev.avgResponseTime * prev.totalQueries + responseTime) / totalQueries;
        const avgContextSize = (prev.avgContextSize * prev.totalQueries + contextSize) / totalQueries;
        const deepTraversalCount = newHistory.filter((q: any) => q.useDeepTraversal).length;
        const deepTraversalUsage = (deepTraversalCount / newHistory.length) * 100;

        return {
          totalQueries,
          avgResponseTime,
          avgContextSize,
          deepTraversalUsage,
          queriesHistory: newHistory
        };
      });

      setSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { 
              ...session, 
              messages: [...updatedMessages, { 
                role: "assistant", 
                content: botMsg,
                citations,
                comparison,
                traversal_info: traversalInfo,
                resolved_references: resolvedReferences,
                hallucination_warning: hallucinationWarning
              }] 
            }
          : session
      ));

    } catch (error) {
      const err = error as { response?: { data?: ApiError | { detail: any } } };
      let message = "응답 생성 실패";
      
      if (err.response?.data) {
        const data = err.response.data;
        if (typeof data.detail === 'string') {
          message = data.detail;
        } else if (Array.isArray(data.detail)) {
          message = data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ');
        } else {
          message = JSON.stringify(data.detail);
        }
      }
      
      setSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { 
              ...session, 
              messages: [...updatedMessages, { 
                role: "assistant", 
                content: `❌ 오류: ${message}` 
              }] 
            }
          : session
      ));
      
      toast.error(message);
    } finally {
      setIsGenerating(false);
    }
  };

  return {
    input,
    setInput,
    isGenerating,
    handleSendMessage,
  };
}
