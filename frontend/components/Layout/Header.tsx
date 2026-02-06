import { PanelLeft, Upload, Loader2, Settings, Download, FolderTree, Activity } from "lucide-react";
import type { ChatSession } from "@/lib/types";

interface HeaderProps {
  isSidebarOpen: boolean;
  currentSessionId: string | null;
  currentSession: ChatSession | undefined;
  isUploading: boolean;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  showPerformance: boolean;
  showSettings: boolean;
  onToggleSidebar: () => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onTogglePerformance: () => void;
  onToggleSettings: () => void;
  onExport: () => void;
  onLoadTree: () => void;
  t: any;
}

export default function Header({ 
  isSidebarOpen,
  currentSessionId,
  currentSession,
  isUploading,
  fileInputRef,
  showPerformance,
  showSettings,
  onToggleSidebar,
  onFileUpload,
  onTogglePerformance,
  onToggleSettings,
  onExport,
  onLoadTree,
  t 
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-white z-10">
      <div className="flex items-center gap-2">
        {!isSidebarOpen && (
          <button 
            onClick={onToggleSidebar}
            className="p-2 hover:bg-slate-100 rounded-full text-slate-500 mr-2"
            aria-label={t.openSidebar}
          >
            <PanelLeft size={20} />
          </button>
        )}
        <h1 className="text-lg font-semibold text-slate-700 flex items-center gap-2">
          TreeRAG <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">AI</span>
        </h1>
      </div>

      {!currentSessionId && (
        <div className="flex items-center gap-3">
          <label className="cursor-pointer flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors">
            {isUploading ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
            {isUploading ? t.processing : t.uploadPdf}
            <input 
              ref={fileInputRef}
              type="file" 
              accept=".pdf"
              multiple
              className="hidden" 
              onChange={onFileUpload}
              disabled={isUploading}
            />
          </label>
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          onClick={onTogglePerformance}
          className="flex items-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 text-sm rounded-lg transition-colors"
          title={t.performance}
        >
          <Activity size={16} />
          {t.performance}
        </button>

        <button
          onClick={onToggleSettings}
          className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm rounded-lg transition-colors"
          title={t.settings}
        >
          <Settings size={16} />
          {t.settings}
        </button>
        
        {currentSessionId && currentSession && (
          <>
            <button
              onClick={onExport}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-100 hover:bg-emerald-200 text-emerald-700 text-sm rounded-lg transition-colors"
              title={t.export}
            >
              <Download size={16} />
              {t.export}
            </button>
            <button
              onClick={onLoadTree}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm rounded-lg transition-colors"
              title={t.treeStructure}
            >
              <FolderTree size={16} />
              {t.treeStructure}
            </button>
          </>
        )}
      </div>
    </header>
  );
}
