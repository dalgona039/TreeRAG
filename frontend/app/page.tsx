"use client";

import { useEffect } from "react";
import toast, { Toaster } from "react-hot-toast";
import { UI_TEXT } from "@/constants/ui-text";
import { 
  useSessionStore, 
  useSettingsStore, 
  useUIStore, 
  usePerformanceStore, 
  useChatStore 
} from "@/stores";
import { useUpload } from "@/hooks/useUpload";
import { useTree } from "@/hooks/useTree";
import Sidebar from "@/components/Sidebar/Sidebar";
import Header from "@/components/Layout/Header";
import WelcomeScreen from "@/components/ui/WelcomeScreen";
import UploadProgress from "@/components/ui/UploadProgress";
import ChatPanel from "@/components/Chat/ChatPanel";
import DocumentPanel from "@/components/Document/DocumentPanel";
import SettingsPanel from "@/components/Settings/SettingsPanel";
import PerformancePanel from "@/components/Settings/PerformancePanel";
import PdfViewer from "@/components/Layout/PdfViewer";

export default function Home() {
  // ===== Zustand Stores (No Prop Drilling!) =====
  
  // Session store
  const sessions = useSessionStore(state => state.sessions);
  const currentSessionId = useSessionStore(state => state.currentSessionId);
  const setSessions = useSessionStore(state => state.setSessions);
  const setCurrentSessionId = useSessionStore(state => state.setCurrentSessionId);
  const createNewSession = useSessionStore(state => state.createNewSession);
  const deleteSession = useSessionStore(state => state.deleteSession);
  const currentSession = sessions.find(s => s.id === currentSessionId);
  
  // Settings store
  const language = useSettingsStore(state => state.language);
  const domainTemplate = useSettingsStore(state => state.domainTemplate);
  const useDeepTraversal = useSettingsStore(state => state.useDeepTraversal);
  const maxDepth = useSettingsStore(state => state.maxDepth);
  const maxBranches = useSettingsStore(state => state.maxBranches);
  const setLanguage = useSettingsStore(state => state.setLanguage);
  const setDomainTemplate = useSettingsStore(state => state.setDomainTemplate);
  const setUseDeepTraversal = useSettingsStore(state => state.setUseDeepTraversal);
  const setMaxDepth = useSettingsStore(state => state.setMaxDepth);
  const setMaxBranches = useSettingsStore(state => state.setMaxBranches);
  
  // UI store
  const isSidebarOpen = useUIStore(state => state.isSidebarOpen);
  const showSettings = useUIStore(state => state.showSettings);
  const showPerformance = useUIStore(state => state.showPerformance);
  const showPdfViewer = useUIStore(state => state.showPdfViewer);
  const pdfFile = useUIStore(state => state.pdfFile);
  const pdfPage = useUIStore(state => state.pdfPage);
  const copiedId = useUIStore(state => state.copiedId);
  const setSidebarOpen = useUIStore(state => state.setSidebarOpen);
  const setShowSettings = useUIStore(state => state.setShowSettings);
  const setShowPerformance = useUIStore(state => state.setShowPerformance);
  const setShowPdfViewer = useUIStore(state => state.setShowPdfViewer);
  const openPdf = useUIStore(state => state.openPdf);
  const setCopiedId = useUIStore(state => state.setCopiedId);
  
  // Wrapper for language change (component expects string, store uses union type)
  const handleLanguageChange = (value: string) => {
    if (value === 'ko' || value === 'en' || value === 'ja') {
      setLanguage(value);
    }
  };
  
  // Performance store - select fields individually to avoid infinite loop
  const totalQueries = usePerformanceStore(state => state.totalQueries);
  const avgResponseTime = usePerformanceStore(state => state.avgResponseTime);
  const avgContextSize = usePerformanceStore(state => state.avgContextSize);
  const deepTraversalUsage = usePerformanceStore(state => state.deepTraversalUsage);
  const queriesHistory = usePerformanceStore(state => state.queriesHistory);
  
  const performanceMetrics = {
    totalQueries,
    avgResponseTime,
    avgContextSize,
    deepTraversalUsage,
    queriesHistory,
  };
  
  // Chat store
  const input = useChatStore(state => state.input);
  const isGenerating = useChatStore(state => state.isGenerating);
  const setInput = useChatStore(state => state.setInput);
  const sendMessage = useChatStore(state => state.sendMessage);
  
  // ===== Legacy Hooks (to be migrated later) =====
  
  const { 
    isUploading, 
    uploadProgress, 
    fileInputRef, 
    handleFileUploadAndIndex,
    loadExistingIndex
  } = useUpload(setSessions, setCurrentSessionId);

  const {
    showTree,
    setShowTree,
    treeData,
    expandedNodes,
    selectedNode,
    setSelectedNode,
    loadTreeStructure,
    handleNodeClick
  } = useTree();
  
  // ===== Computed Values =====
  
  const t = UI_TEXT[language as keyof typeof UI_TEXT] || UI_TEXT.ko;
  
  // ===== Effects =====
  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        createNewSession();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [createNewSession]);

  // ===== Event Handlers =====

  const handleCitationClick = (citation: string) => {
    const match = citation.match(/(.+?),\s*p\.(\d+)/);
    if (match) {
      const [_, docName, pageNum] = match;
      const citationDocName = docName.trim();
      
      let filename = `${citationDocName}.pdf`;
      
      if (currentSession?.originalFilenames && currentSession.originalFilenames.length > 0) {
        const matchingFile = currentSession.originalFilenames.find(originalFilename => {
          const nameWithoutExt = originalFilename.replace('.pdf', '');
          return nameWithoutExt === citationDocName || 
                 nameWithoutExt.includes(citationDocName) || 
                 citationDocName.includes(nameWithoutExt);
        });
        
        if (matchingFile) {
          filename = matchingFile;
        } else if (currentSession.originalFilenames.length === 1) {
          filename = currentSession.originalFilenames[0];
        }
      }
      
      openPdf(filename, parseInt(pageNum));
      toast.success(`${t.pdfOpen}: ${filename} (p.${pageNum})`);
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      toast.success(t.copiedToClipboard);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      toast.error(t.copyFailed);
    }
  };

  const exportToMarkdown = (session: typeof currentSession) => {
    if (!session) return;
    
    let markdown = `# ${session.title}\n\n`;
    markdown += `**생성일:** ${session.createdAt.toLocaleString('ko-KR')}\n\n`;
    markdown += `**문서:** ${session.indexFiles.map(f => f.replace('_index.json', '')).join(', ')}\n\n`;
    markdown += `---\n\n`;

    session.messages.forEach((msg, idx) => {
      if (msg.role === 'user') {
        markdown += `## 질문 ${Math.floor((idx + 1) / 2)}\n\n`;
        markdown += `> ${msg.content}\n\n`;
      } else if (msg.role === 'assistant') {
        markdown += `### 답변\n\n`;
        markdown += `${msg.content}\n\n`;
        
        if (msg.citations && msg.citations.length > 0) {
          markdown += `**출처:**\n`;
          msg.citations.forEach((citation: string) => {
            markdown += `- ${citation}\n`;
          });
          markdown += `\n`;
        }
        
        if (msg.resolved_references && msg.resolved_references.length > 0) {
          markdown += `**Cross-reference 해결됨:**\n`;
          msg.resolved_references.forEach((ref: { title: string; page_ref?: string }) => {
            markdown += `- ${ref.title}`;
            if (ref.page_ref) markdown += ` (${ref.page_ref})`;
            markdown += `\n`;
          });
          markdown += `\n`;
        }
        
        if (msg.traversal_info && msg.traversal_info.used_deep_traversal) {
          markdown += `**Deep Traversal 통계:**\n`;
          markdown += `- Nodes Visited: ${msg.traversal_info.nodes_visited.length}\n`;
          markdown += `- Nodes Selected: ${msg.traversal_info.nodes_selected.length}\n`;
          markdown += `- Max Depth: ${msg.traversal_info.max_depth}\n`;
          markdown += `- Max Branches: ${msg.traversal_info.max_branches}\n\n`;
        }
        
        if (msg.comparison && msg.comparison.has_comparison) {
          markdown += `**문서 비교 분석**\n\n`;
          markdown += `비교 대상: ${msg.comparison.documents_compared.join(' ↔ ')}\n\n`;
          if (msg.comparison.commonalities) {
            markdown += `**공통점:**\n${msg.comparison.commonalities}\n\n`;
          }
          if (msg.comparison.differences) {
            markdown += `**차이점:**\n${msg.comparison.differences}\n\n`;
          }
        }
        
        markdown += `---\n\n`;
      }
    });

    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${session.title.replace(/[^a-zA-Z0-9가-힣\s]/g, '_')}_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success(t.markdownSaved);
  };

  const handleDeleteSession = (sessionId: string) => {
    deleteSession(sessionId);
    toast.success(t.sessionDeleted);
  };

  // ===== Render =====

  return (
    <div className="flex h-screen bg-white font-sans text-slate-800 overflow-hidden">
      <Toaster position="top-center" />
      
      <Sidebar
        isOpen={isSidebarOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onClose={() => setSidebarOpen(false)}
        onNewSession={createNewSession}
        onSelectSession={setCurrentSessionId}
        onDeleteSession={handleDeleteSession}
        onLoadExistingIndex={(indexFilename) => loadExistingIndex(indexFilename, t)}
        t={t}
      />

      <main className="flex-1 flex flex-col h-full relative">
        <Header
          isSidebarOpen={isSidebarOpen}
          currentSessionId={currentSessionId}
          currentSession={currentSession}
          isUploading={isUploading}
          fileInputRef={fileInputRef}
          showPerformance={showPerformance}
          showSettings={showSettings}
          onToggleSidebar={() => setSidebarOpen(true)}
          onFileUpload={(e) => handleFileUploadAndIndex(e, t)}
          onTogglePerformance={() => setShowPerformance(!showPerformance)}
          onToggleSettings={() => setShowSettings(!showSettings)}
          onExport={() => exportToMarkdown(currentSession)}
          onLoadTree={() => currentSession && loadTreeStructure(currentSession.indexFiles[0], t)}
          t={t}
        />

        {uploadProgress && <UploadProgress uploadProgress={uploadProgress} t={t} />}

        <PerformancePanel 
          showPerformance={showPerformance} 
          performanceMetrics={performanceMetrics}
          t={t}
        />

        <SettingsPanel
          showSettings={showSettings}
          domainTemplate={domainTemplate}
          language={language}
          useDeepTraversal={useDeepTraversal}
          maxDepth={maxDepth}
          maxBranches={maxBranches}
          onDomainChange={setDomainTemplate}
          onLanguageChange={handleLanguageChange}
          onDeepTraversalChange={setUseDeepTraversal}
          onMaxDepthChange={setMaxDepth}
          onMaxBranchesChange={setMaxBranches}
          t={t}
        />

        {!currentSessionId ? (
          <WelcomeScreen 
            t={t} 
            onLoadExistingIndex={(indexFilename) => loadExistingIndex(indexFilename, t)}
          />
        ) : (
          <ChatPanel
            currentSessionId={currentSessionId}
            messages={currentSession?.messages || []}
            input={input}
            isGenerating={isGenerating}
            selectedNode={selectedNode}
            onInputChange={setInput}
            onSendMessage={sendMessage}
            onCitationClick={handleCitationClick}
            onCopy={copyToClipboard}
            onDeselectNode={() => {
              setSelectedNode(null);
              toast.success(t.sectionDeselected);
            }}
            copiedId={copiedId}
            t={t}
          />
        )}
      </main>

      <DocumentPanel
        showTree={showTree}
        treeData={treeData}
        expandedNodes={expandedNodes}
        selectedNode={selectedNode}
        onClose={() => {
          setShowTree(false);
          setSelectedNode(null);
        }}
        onNodeClick={(node, hasChildren, e) => handleNodeClick(node, hasChildren, e, setInput, t)}
        t={t}
      />

      <PdfViewer
        showPdfViewer={showPdfViewer}
        pdfFile={pdfFile}
        pdfPage={pdfPage}
        onClose={() => setShowPdfViewer(false)}
      />
    </div>
  );
}
