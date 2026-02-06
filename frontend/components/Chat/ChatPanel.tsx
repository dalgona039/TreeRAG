import { Send } from "lucide-react";
import MessageList from "../Chat/MessageList";
import type { Message, TreeNode } from "@/lib/types";

interface ChatPanelProps {
  currentSessionId: string | null;
  messages: Message[];
  input: string;
  isGenerating: boolean;
  selectedNode: TreeNode | null;
  onInputChange: (value: string) => void;
  onSendMessage: () => void;
  onCitationClick: (citation: string) => void;
  onCopy: (text: string, id: string) => void;
  onDeselectNode: () => void;
  copiedId: string | null;
  t: any;
}

export default function ChatPanel({ 
  currentSessionId,
  messages,
  input,
  isGenerating,
  selectedNode,
  onInputChange,
  onSendMessage,
  onCitationClick,
  onCopy,
  onDeselectNode,
  copiedId,
  t 
}: ChatPanelProps) {
  return (
    <>
      <MessageList
        messages={messages}
        isGenerating={isGenerating}
        onCitationClick={onCitationClick}
        onCopy={onCopy}
        copiedId={copiedId}
        t={t}
      />

      {currentSessionId && (
        <div className="bg-white p-4 md:pb-6 border-t border-slate-100">
          {selectedNode && (
            <div className="max-w-3xl mx-auto mb-3 flex items-center gap-2 text-xs bg-indigo-50 px-4 py-2 rounded-lg border border-indigo-200">
              <span className="text-indigo-700">📌 {t.selectedSection}:</span>
              <span className="font-medium text-indigo-900">{selectedNode.title}</span>
              {selectedNode.page_ref && (
                <span className="text-indigo-600">(p.{selectedNode.page_ref})</span>
              )}
              <button
                onClick={onDeselectNode}
                className="ml-auto text-indigo-600 hover:text-indigo-800"
                aria-label={t.sectionDeselected}
              >
                ✕
              </button>
            </div>
          )}
          <div className="max-w-3xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !isGenerating && onSendMessage()}
              placeholder={selectedNode ? `"${selectedNode.title}" ${t.sectionQuestion}` : t.typeMessage}
              disabled={isGenerating}
              className="w-full bg-[#f0f4f9] hover:bg-[#e9eef6] focus:bg-white border-2 border-transparent focus:border-indigo-200 rounded-full pl-6 pr-14 py-4 text-slate-700 placeholder:text-slate-400 focus:outline-none transition-all shadow-sm"
              aria-label={t.typeMessage}
            />
            <button 
              onClick={onSendMessage}
              disabled={!input.trim() || isGenerating}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
              aria-label={t.send}
            >
              <Send size={18} />
            </button>
          </div>
          <div className="text-center mt-2 text-xs text-slate-400">
            {t.disclaimer}
          </div>
        </div>
      )}
    </>
  );
}
