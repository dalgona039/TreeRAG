"use client";

import { useState, useEffect } from "react";
import toast, { Toaster } from "react-hot-toast";
import { UI_TEXT } from "@/constants/ui-text";
import { useSessions } from "@/hooks/useSessions";
import { useUpload } from "@/hooks/useUpload";
import { useChat } from "@/hooks/useChat";
import { useTree } from "@/hooks/useTree";
import { usePerformance } from "@/hooks/usePerformance";
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
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [pdfFile, setPdfFile] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [showSettings, setShowSettings] = useState(false);
  const [showPerformance, setShowPerformance] = useState(false);
  const [useDeepTraversal, setUseDeepTraversal] = useState(true);
  const [maxDepth, setMaxDepth] = useState(5);
  const [maxBranches, setMaxBranches] = useState(3);
  const [domainTemplate, setDomainTemplate] = useState("general");
  const [language, setLanguage] = useState("ko");

  const { 
    sessions, 
    setSessions, 
    currentSessionId, 
    setCurrentSessionId, 
    currentSession,
    createNewSession, 
    deleteSession 
  } = useSessions();

  const { performanceMetrics, setPerformanceMetrics } = usePerformance();

  const { 
    isUploading, 
    uploadProgress, 
    fileInputRef, 
    handleFileUploadAndIndex 
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

  const { 
    input, 
    setInput, 
    isGenerating, 
    handleSendMessage 
  } = useChat(
    sessions,
    setSessions,
    currentSessionId,
    currentSession,
    selectedNode,
    useDeepTraversal,
    maxDepth,
    maxBranches,
    domainTemplate,
    language,
    setPerformanceMetrics
  );

  const t = UI_TEXT[language as keyof typeof UI_TEXT] || UI_TEXT.ko;

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

  const handleCitationClick = (citation: string) => {
    const match = citation.match(/(.+?),\s*p\.(\d+)/);
    if (match) {
      const [_, docName, pageNum] = match;
      const citationDocName = docName.trim();
      
      console.log('[Citation Click] 원본 citation:', citation);
      console.log('[Citation Click] 추출된 문서명:', citationDocName);
      console.log('[Citation Click] 현재 세션:', currentSession);
      console.log('[Citation Click] originalFilenames:', currentSession?.originalFilenames);
      
      let filename = `${citationDocName}.pdf`;
      
      if (currentSession?.originalFilenames && currentSession.originalFilenames.length > 0) {
        console.log('[Citation Click] originalFilenames 사용 중...');
        
        const matchingFile = currentSession.originalFilenames.find(originalFilename => {
          const nameWithoutExt = originalFilename.replace('.pdf', '');
          const matches = nameWithoutExt === citationDocName || 
                 nameWithoutExt.includes(citationDocName) || 
                 citationDocName.includes(nameWithoutExt);
          console.log(`[Citation Click] 비교: "${nameWithoutExt}" vs "${citationDocName}" => ${matches}`);
          return matches;
        });
        
        if (matchingFile) {
          console.log('[Citation Click] 매칭된 파일:', matchingFile);
          filename = matchingFile;
        } else {
          console.log('[Citation Click] 매칭 실패, 단일 문서 체크...');
          if (currentSession.originalFilenames.length === 1) {
            filename = currentSession.originalFilenames[0];
            console.log('[Citation Click] 단일 문서 사용:', filename);
          }
        }
      } else {
        console.log('[Citation Click] originalFilenames 없음, 기본 파일명 사용');
      }
      
      console.log('[Citation Click] 최종 파일명:', filename);
      console.log('[Citation Click] 요청 URL:', `/api/pdf/${encodeURIComponent(filename)}`);
      
      setPdfFile(filename);
      setPdfPage(parseInt(pageNum));
      setShowPdfViewer(true);
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
          msg.citations.forEach(citation => {
            markdown += `- ${citation}\n`;
          });
          markdown += `\n`;
        }
        
        if (msg.resolved_references && msg.resolved_references.length > 0) {
          markdown += `**Cross-reference 해결됨:**\n`;
          msg.resolved_references.forEach(ref => {
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

  return (
    <div className="flex h-screen bg-white font-sans text-slate-800 overflow-hidden">
      <Toaster position="top-center" />
      
      <Sidebar
        isOpen={isSidebarOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onClose={() => setIsSidebarOpen(false)}
        onNewSession={createNewSession}
        onSelectSession={setCurrentSessionId}
        onDeleteSession={deleteSession}
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
          onToggleSidebar={() => setIsSidebarOpen(true)}
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
          onLanguageChange={setLanguage}
          onDeepTraversalChange={setUseDeepTraversal}
          onMaxDepthChange={setMaxDepth}
          onMaxBranchesChange={setMaxBranches}
          t={t}
        />

        {!currentSessionId ? (
          <WelcomeScreen t={t} />
        ) : (
          <ChatPanel
            currentSessionId={currentSessionId}
            messages={currentSession?.messages || []}
            input={input}
            isGenerating={isGenerating}
            selectedNode={selectedNode}
            onInputChange={setInput}
            onSendMessage={handleSendMessage}
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