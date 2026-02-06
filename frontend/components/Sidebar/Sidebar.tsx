import { useState } from "react";
import { Plus, PanelLeftClose, Search, X } from "lucide-react";
import SessionItem from "./SessionItem";
import type { ChatSession } from "@/lib/types";

interface SidebarProps {
  isOpen: boolean;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onClose: () => void;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  t: any;
}

export default function Sidebar({ 
  isOpen, 
  sessions, 
  currentSessionId, 
  onClose, 
  onNewSession, 
  onSelectSession, 
  onDeleteSession,
  t 
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredSessions = sessions.filter(session => {
    if (!searchQuery.trim()) return true;
    const query = searchQuery.toLowerCase();
    if (session.title.toLowerCase().includes(query)) return true;
    return session.messages.some(msg => 
      msg.content.toLowerCase().includes(query)
    );
  });

  return (
    <aside 
      className={`${isOpen ? "w-72" : "w-0"} bg-[#f0f4f9] transition-all duration-300 flex flex-col border-r border-slate-200 overflow-hidden`}
    >
      <div className="p-4 flex items-center justify-between">
        <button 
          onClick={onClose}
          className="p-2 hover:bg-slate-200 rounded-full text-slate-500"
          aria-label={t.closeSidebar}
        >
          <PanelLeftClose size={20} />
        </button>
      </div>

      <div className="px-4 mb-6">
        <button 
          onClick={onNewSession}
          className="flex items-center gap-3 bg-[#dde3ea] hover:bg-[#d0dbe7] text-slate-700 px-4 py-3 rounded-xl w-full transition-colors font-medium text-sm"
          title={`${t.newChat} (Ctrl+K)`}
        >
          <Plus size={18} />
          {t.newChat}
        </button>
      </div>

      <div className="px-4 mb-4">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t.searchPlaceholder}
            className="w-full pl-9 pr-3 py-2 bg-white border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-2">
        <div className="text-xs font-semibold text-slate-500 px-4 mb-2">
          {searchQuery ? t.searchResults : t.recentHistory}
        </div>
        {filteredSessions.map((session) => (
          <SessionItem
            key={session.id}
            session={session}
            isActive={currentSessionId === session.id}
            onSelect={() => onSelectSession(session.id)}
            onDelete={(e) => onDeleteSession(session.id, e)}
            t={t}
          />
        ))}
        
        {sessions.length === 0 && (
          <div className="text-center text-slate-400 text-xs mt-10">
            {t.noHistory}
          </div>
        )}
        
        {sessions.length > 0 && searchQuery && filteredSessions.length === 0 && (
          <div className="text-center text-slate-400 text-xs mt-10">
            {t.noSearchResults}
          </div>
        )}
      </div>
    </aside>
  );
}
