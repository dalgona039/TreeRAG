import { useRef, useEffect } from "react";
import { Loader2 } from "lucide-react";
import MessageItem from "./MessageItem";
import type { Message } from "@/lib/types";

interface MessageListProps {
  messages: Message[];
  isGenerating: boolean;
  onCitationClick: (citation: string) => void;
  onCopy: (text: string, id: string) => void;
  copiedId: string | null;
  t: any;
}

export default function MessageList({ 
  messages, 
  isGenerating, 
  onCitationClick, 
  onCopy, 
  copiedId,
  t 
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth bg-white">
      {messages.map((msg, idx) => (
        <MessageItem
          key={idx}
          message={msg}
          index={idx}
          onCitationClick={onCitationClick}
          onCopy={onCopy}
          copiedId={copiedId}
          t={t}
        />
      ))}
      
      {isGenerating && (
        <div className="flex gap-4 max-w-4xl mx-auto">
          <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
            <Loader2 size={18} className="animate-spin text-indigo-600" />
          </div>
          <div className="px-5 py-3 bg-white text-slate-500 text-sm">
            {t.analyzing}
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
