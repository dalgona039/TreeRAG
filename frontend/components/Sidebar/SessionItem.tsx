import { MessageSquare, Trash2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { ko } from "date-fns/locale";
import type { ChatSession } from "@/lib/types";

interface SessionItemProps {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
  t: {
    deleteSession: string;
  };
}

export default function SessionItem({ 
  session, 
  isActive, 
  onSelect, 
  onDelete,
  t 
}: SessionItemProps) {
  return (
    <div
      className={`group relative w-full text-left flex items-center gap-3 px-4 py-2 rounded-full text-sm mb-1 transition-colors ${
        isActive 
          ? "bg-[#c4d7ed] text-slate-900 font-medium" 
          : "hover:bg-[#e0e5eb] text-slate-600"
      }`}
    >
      <button
        onClick={onSelect}
        className="flex items-center gap-3 flex-1 min-w-0"
      >
        <MessageSquare size={16} className="flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="truncate">{session.title}</div>
          <div className="text-xs text-slate-400">
            {formatDistanceToNow(session.createdAt, { addSuffix: true, locale: ko })}
          </div>
        </div>
      </button>
      <button
        onClick={onDelete}
        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded-full transition-opacity"
        aria-label={t.deleteSession}
      >
        <Trash2 size={14} className="text-red-600" />
      </button>
    </div>
  );
}
