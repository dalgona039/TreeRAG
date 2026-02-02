"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import toast, { Toaster } from "react-hot-toast";
import ReactMarkdown from "react-markdown";
import { formatDistanceToNow } from "date-fns";
import { ko } from "date-fns/locale";
import { 
  Upload, FileText, Send, Bot, User, Loader2, 
  Plus, MessageSquare, PanelLeftClose, PanelLeft,
  Trash2, Copy, Check, ChevronRight, ChevronDown, FolderTree,
  Settings, X, Download
} from "lucide-react";

type TreeNode = {
  id: string;
  title: string;
  summary?: string;
  page_ref?: string;
  children?: TreeNode[];
};

type TreeData = {
  document_name: string;
  tree: TreeNode;
};

type ComparisonResult = {
  has_comparison: boolean;
  documents_compared: string[];
  commonalities?: string;
  differences?: string;
};

type TraversalInfo = {
  used_deep_traversal: boolean;
  nodes_visited: string[];
  nodes_selected: Array<{
    document: string;
    title: string;
    page_ref: string;
  }>;
  max_depth: number;
  max_branches: number;
};

type ResolvedReference = {
  title: string;
  page_ref?: string;
  summary?: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  comparison?: ComparisonResult;
  traversal_info?: TraversalInfo;
  resolved_references?: ResolvedReference[];
};

type ChatSession = {
  id: string;
  title: string;
  indexFiles: string[];
  messages: Message[];
  createdAt: Date;
};

type ApiError = {
  detail: string;
};

const API_BASE_URL = "http://localhost:8000/api";
const STORAGE_KEY = "treerag-sessions";

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showTree, setShowTree] = useState(false);
  const [treeData, setTreeData] = useState<TreeData | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [useDeepTraversal, setUseDeepTraversal] = useState(true);
  const [maxDepth, setMaxDepth] = useState(5);
  const [maxBranches, setMaxBranches] = useState(3);
  const [showSettings, setShowSettings] = useState(false);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [pdfFile, setPdfFile] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [sessions, currentSessionId]);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSessions(parsed.map((s: ChatSession) => ({
          ...s,
          createdAt: new Date(s.createdAt)
        })));
      } catch (error) {
        console.error("Failed to load sessions:", error);
      }
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }
  }, [sessions]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        createNewSession();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const currentSession = sessions.find(s => s.id === currentSessionId);
  const currentMessages = currentSession?.messages || [];

  const createNewSession = () => {
    setCurrentSessionId(null);
    setInput("");
  };

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
    }
    toast.success("세션이 삭제되었습니다");
  };

  const handleFileUploadAndIndex = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const files = Array.from(e.target.files);
    const loadingToast = toast.loading(`${files.length}개의 파일을 업로드하고 분석 중입니다...`);
    
    try {
      setIsUploading(true);

      const indexFiles: string[] = [];
      const docNames: string[] = [];

      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        
        await axios.post(`${API_BASE_URL}/upload`, formData);

        const indexRes = await axios.post(`${API_BASE_URL}/index`, {
          filename: file.name,
        });
        
        indexFiles.push(indexRes.data.index_file);
        docNames.push(file.name.replace('.pdf', ''));
      }

      const sessionTitle = files.length === 1 
        ? docNames[0] 
        : `${docNames[0]} 외 ${files.length - 1}건`;

      const newSession: ChatSession = {
        id: Date.now().toString(),
        title: sessionTitle,
        indexFiles: indexFiles,
        messages: [{ 
          role: "assistant", 
          content: `반갑습니다! ${files.length}개 문서(${docNames.join(", ")})에 대한 분석 준비가 완료되었습니다. 무엇이든 물어보세요.` 
        }],
        createdAt: new Date(),
      };

      setSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      
      toast.success("분석이 완료되었습니다!", { id: loadingToast });
    } catch (error) {
      const err = error as { response?: { data?: ApiError } };
      const message = err.response?.data?.detail || "업로드/분석 실패";
      toast.error(message, { id: loadingToast });
      console.error(error);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

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

    try {
      const requestBody: any = {
        question: userMsg,
        index_filenames: currentSession.indexFiles,
        use_deep_traversal: useDeepTraversal,
        max_depth: maxDepth,
        max_branches: maxBranches,
      };
      
      if (selectedNode) {
        requestBody.node_context = {
          id: selectedNode.id,
          title: selectedNode.title,
          page_ref: selectedNode.page_ref,
          summary: selectedNode.summary,
        };
      }
      
      const res = await axios.post(`${API_BASE_URL}/chat`, requestBody);
      
      const botMsg = res.data.answer;
      const citations = res.data.citations || [];
      const comparison = res.data.comparison || null;
      const traversalInfo = res.data.traversal_info || null;
      const resolvedReferences = res.data.resolved_references || null;

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
                resolved_references: resolvedReferences
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

  const loadTreeStructure = async (indexFilename: string) => {
    try {
      const res = await axios.get(`${API_BASE_URL}/tree/${indexFilename}`);
      setTreeData(res.data);
      setShowTree(true);
      setExpandedNodes(new Set([res.data.tree.id]));
      toast.success(`트리 로드 완료: ${res.data.document_name}`);
    } catch (error) {
      toast.error("트리 로드 실패");
      console.error(error);
    }
  };

  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const handleNodeClick = (node: TreeNode, hasChildren: boolean, e: React.MouseEvent) => {
    if (hasChildren) {
      toggleNode(node.id);
    }
    
    // Shift + 클릭으로 노드 선택 및 질문 생성
    if (e.shiftKey) {
      e.stopPropagation();
      setSelectedNode(node);
      
      const question = `"${node.title}" 섹션에 대해 자세히 설명해주세요.${node.page_ref ? ` (페이지 ${node.page_ref})` : ''}`;
      setInput(question);
      toast.success(`노드 선택됨: ${node.title}`);
    }
  };

  const handleCitationClick = (citation: string) => {
    const match = citation.match(/(.+?),\s*p\.(\d+)/);
    if (match) {
      const [_, docName, pageNum] = match;
      const filename = `${docName.trim()}.pdf`;
      setPdfFile(filename);
      setPdfPage(parseInt(pageNum));
      setShowPdfViewer(true);
      toast.success(`PDF 열기: ${filename} (p.${pageNum})`);
    }
  };

  const exportToMarkdown = (session: ChatSession) => {
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
    toast.success('Markdown 파일로 저장되었습니다');
  };

  const renderTreeNode = (node: TreeNode, level: number = 0): JSX.Element => {
    const isExpanded = expandedNodes.has(node.id);
    const hasChildren = node.children && node.children.length > 0;
    const isSelected = selectedNode?.id === node.id;
    
    return (
      <div key={node.id} className="mb-1">
        <div 
          className={`flex items-start gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
            level > 0 ? 'ml-' + (level * 4) : ''
          } ${
            isSelected ? 'bg-indigo-100 border border-indigo-300' : 'hover:bg-slate-50'
          }`}
          onClick={(e) => handleNodeClick(node, hasChildren, e)}
          title="클릭: 펼치기/접기 | Shift+클릭: 이 섹션 질문하기"
        >
          {hasChildren ? (
            isExpanded ? <ChevronDown size={16} className="mt-1 text-slate-600" /> : <ChevronRight size={16} className="mt-1 text-slate-600" />
          ) : (
            <div className="w-4" />
          )}
          <div className="flex-1 min-w-0">
            <div className={`font-medium text-sm ${
              isSelected ? 'text-indigo-800' : 'text-slate-800'
            }`}>{node.title}</div>
            {node.page_ref && (
              <div className="text-xs text-indigo-600 mt-0.5">📄 p.{node.page_ref}</div>
            )}
            {node.summary && isExpanded && (
              <div className="text-xs text-slate-600 mt-1 leading-relaxed">{node.summary}</div>
            )}
          </div>
        </div>
        {isExpanded && hasChildren && (
          <div className="ml-2">
            {node.children!.map(child => renderTreeNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      toast.success("클립보드에 복사되었습니다");
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      toast.error("복사 실패");
    }
  };

  return (
    <div className="flex h-screen bg-white font-sans text-slate-800 overflow-hidden">
      <Toaster position="top-center" />
      
      <aside 
        className={`${isSidebarOpen ? "w-72" : "w-0"} bg-[#f0f4f9] transition-all duration-300 flex flex-col border-r border-slate-200 overflow-hidden`}
      >
        <div className="p-4 flex items-center justify-between">
          <button 
            onClick={() => setIsSidebarOpen(false)}
            className="p-2 hover:bg-slate-200 rounded-full text-slate-500"
            aria-label="사이드바 닫기"
          >
            <PanelLeftClose size={20} />
          </button>
        </div>

        <div className="px-4 mb-6">
          <button 
            onClick={createNewSession}
            className="flex items-center gap-3 bg-[#dde3ea] hover:bg-[#d0dbe7] text-slate-700 px-4 py-3 rounded-xl w-full transition-colors font-medium text-sm"
            title="새 세션 (Ctrl+K)"
          >
            <Plus size={18} />
            새로운 분석 시작
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2">
          <div className="text-xs font-semibold text-slate-500 px-4 mb-2">최근 기록</div>
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group relative w-full text-left flex items-center gap-3 px-4 py-2 rounded-full text-sm mb-1 transition-colors ${
                currentSessionId === session.id 
                  ? "bg-[#c4d7ed] text-slate-900 font-medium" 
                  : "hover:bg-[#e0e5eb] text-slate-600"
              }`}
            >
              <button
                onClick={() => setCurrentSessionId(session.id)}
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
                onClick={(e) => deleteSession(session.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded-full transition-opacity"
                aria-label="세션 삭제"
              >
                <Trash2 size={14} className="text-red-600" />
              </button>
            </div>
          ))}
          
          {sessions.length === 0 && (
            <div className="text-center text-slate-400 text-xs mt-10">
              기록이 없습니다.
            </div>
          )}
        </div>
      </aside>

      <main className="flex-1 flex flex-col h-full relative">
        
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-white z-10">
          <div className="flex items-center gap-2">
            {!isSidebarOpen && (
              <button 
                onClick={() => setIsSidebarOpen(true)}
                className="p-2 hover:bg-slate-100 rounded-full text-slate-500 mr-2"
                aria-label="사이드바 열기"
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
                {isUploading ? "처리 중..." : "PDF 업로드 및 분석"}
                <input 
                  ref={fileInputRef}
                  type="file" 
                  accept=".pdf"
                  multiple
                  className="hidden" 
                  onChange={handleFileUploadAndIndex}
                  disabled={isUploading}
                />
              </label>
            </div>
          )}

          {currentSessionId && currentSession && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => exportToMarkdown(currentSession)}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-100 hover:bg-emerald-200 text-emerald-700 text-sm rounded-lg transition-colors"
                title="대화 내용 다운로드"
              >
                <Download size={16} />
                Export
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm rounded-lg transition-colors"
                title="Traversal 설정"
              >
                <Settings size={16} />
                설정
              </button>
              <button
                onClick={() => loadTreeStructure(currentSession.indexFiles[0])}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm rounded-lg transition-colors"
                title="문서 구조 보기"
              >
                <FolderTree size={16} />
                트리 구조
              </button>
            </div>
          )}
        </header>

        {showSettings && currentSessionId && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-200 p-4">
            <div className="max-w-4xl mx-auto">
              <h3 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                <Settings size={16} className="text-indigo-600" />
                Deep Traversal 설정
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-3 rounded-lg border border-blue-200">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useDeepTraversal}
                      onChange={(e) => setUseDeepTraversal(e.target.checked)}
                      className="w-4 h-4 text-indigo-600 rounded"
                    />
                    <span className="text-sm font-medium text-slate-700">Deep Traversal 사용</span>
                  </label>
                  <p className="text-xs text-slate-500 mt-1 ml-6">
                    {useDeepTraversal ? "트리를 탐색하여 관련 섹션만 선택" : "전체 문서를 사용 (레거시)"}
                  </p>
                </div>

                <div className="bg-white p-3 rounded-lg border border-blue-200">
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    최대 깊이 (Max Depth)
                  </label>
                  <input
                    type="number"
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(Number(e.target.value))}
                    min="1"
                    max="10"
                    disabled={!useDeepTraversal}
                    className="w-full px-3 py-1 border border-slate-300 rounded text-sm disabled:bg-slate-100 disabled:text-slate-400"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    트리 탐색 최대 깊이 (1-10)
                  </p>
                </div>

                <div className="bg-white p-3 rounded-lg border border-blue-200">
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    브랜치 수 (Max Branches)
                  </label>
                  <input
                    type="number"
                    value={maxBranches}
                    onChange={(e) => setMaxBranches(Number(e.target.value))}
                    min="1"
                    max="10"
                    disabled={!useDeepTraversal}
                    className="w-full px-3 py-1 border border-slate-300 rounded text-sm disabled:bg-slate-100 disabled:text-slate-400"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    레벨당 탐색할 자식 노드 수 (1-10)
                  </p>
                </div>
              </div>
              <div className="mt-3 text-xs text-blue-700 bg-blue-100 p-2 rounded">
                💡 <strong>팁:</strong> 깊이와 브랜치 수를 줄이면 응답 속도가 빨라지지만 정보가 제한될 수 있습니다.
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth bg-white">
          {!currentSessionId ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-80 pb-20">
              <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-xl">
                <FileText className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-slate-700 mb-2">TreeRAG</h2>
              <p className="max-w-md text-center text-slate-500">
                PDF 문서를 업로드하면 AI가 자동으로 구조화하여 분석합니다.<br/>
                계층적 트리 구조로 문서를 탐색하고 정확한 답변을 제공합니다.
              </p>
              <p className="text-xs text-slate-400 mt-4">
                단축키: <kbd className="px-2 py-1 bg-slate-100 rounded">Ctrl+K</kbd> 새 세션
              </p>
            </div>
          ) : (
            currentMessages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''}`}>
                
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0 mt-1">
                    <Bot size={18} className="text-indigo-600" />
                  </div>
                )}

                <div className="flex-1 min-w-0">
                  <div 
                    className={`group relative px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm ${
                      msg.role === 'user' 
                        ? "bg-[#e7effe] text-slate-800 rounded-br-none ml-auto max-w-[80%]" 
                        : "bg-white border border-slate-100 text-slate-800 rounded-tl-none"
                    }`}
                  >
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm max-w-none">
                        <ReactMarkdown>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    )}
                    
                    <button
                      onClick={() => copyToClipboard(msg.content, `${idx}`)}
                      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 p-1.5 bg-slate-100 hover:bg-slate-200 rounded transition-opacity"
                      aria-label="복사"
                    >
                      {copiedId === `${idx}` ? <Check size={14} /> : <Copy size={14} />}
                    </button>
                  </div>
                  
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2 ml-1">
                      {msg.citations.map((citation, i) => (
                        <button
                          key={i}
                          onClick={() => handleCitationClick(citation)}
                          className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full border border-indigo-100 hover:bg-indigo-100 cursor-pointer transition-colors"
                        >
                          📎 {citation}
                        </button>
                      ))}
                    </div>
                  )}
                  
                  {msg.comparison && msg.comparison.has_comparison && (
                    <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-xl">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
                          <span className="text-lg">📊</span>
                        </div>
                        <h4 className="font-semibold text-amber-900">문서 비교 분석</h4>
                      </div>
                      
                      <div className="text-sm text-amber-800 mb-2">
                        <strong>비교 대상:</strong> {msg.comparison.documents_compared.join(" ↔ ")}
                      </div>
                      
                      {msg.comparison.commonalities && (
                        <div className="mb-3">
                          <div className="font-medium text-green-700 mb-1">✓ 공통점</div>
                          <div className="text-sm text-gray-700 bg-white p-2 rounded">
                            {msg.comparison.commonalities}
                          </div>
                        </div>
                      )}
                      
                      {msg.comparison.differences && (
                        <div>
                          <div className="font-medium text-red-700 mb-1">⚠ 차이점</div>
                          <div className="text-sm text-gray-700 bg-white p-2 rounded overflow-x-auto">
                            <ReactMarkdown>{msg.comparison.differences}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {msg.resolved_references && msg.resolved_references.length > 0 && (
                    <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-xl">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                          <span className="text-lg">🔗</span>
                        </div>
                        <h4 className="font-semibold text-purple-900">Cross-reference 해결됨</h4>
                      </div>
                      <div className="text-xs text-purple-700 mb-2">
                        질문에서 {msg.resolved_references.length}개의 참조가 감지되어 자동으로 컨텍스트에 추가되었습니다
                      </div>
                      <div className="space-y-2 max-h-40 overflow-y-auto">
                        {msg.resolved_references.map((ref, i) => (
                          <div key={i} className="bg-white p-2 rounded text-sm">
                            <div className="font-medium text-purple-700">{ref.title}</div>
                            {ref.page_ref && (
                              <div className="text-xs text-slate-500 mt-1">페이지: {ref.page_ref}</div>
                            )}
                            {ref.summary && (
                              <div className="text-xs text-slate-600 mt-1 line-clamp-2">{ref.summary}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.traversal_info && msg.traversal_info.used_deep_traversal && (
                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-xl">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                          <span className="text-lg">🌲</span>
                        </div>
                        <h4 className="font-semibold text-blue-900">Deep Traversal 정보</h4>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3 mb-3 text-sm">
                        <div className="bg-white p-2 rounded">
                          <span className="text-blue-600 font-medium">탐색한 노드:</span>
                          <span className="ml-2 text-slate-700">{msg.traversal_info.nodes_visited.length}개</span>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <span className="text-blue-600 font-medium">선택된 노드:</span>
                          <span className="ml-2 text-slate-700">{msg.traversal_info.nodes_selected.length}개</span>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <span className="text-blue-600 font-medium">최대 깊이:</span>
                          <span className="ml-2 text-slate-700">{msg.traversal_info.max_depth}</span>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <span className="text-blue-600 font-medium">브랜치 수:</span>
                          <span className="ml-2 text-slate-700">{msg.traversal_info.max_branches}</span>
                        </div>
                      </div>

                      {msg.traversal_info.nodes_selected.length > 0 && (
                        <div>
                          <div className="font-medium text-blue-700 mb-2 text-sm">선택된 섹션:</div>
                          <div className="space-y-1 max-h-32 overflow-y-auto">
                            {msg.traversal_info.nodes_selected.map((node, i) => (
                              <div key={i} className="text-xs bg-white p-2 rounded flex items-start gap-2">
                                <span className="text-blue-500 flex-shrink-0">•</span>
                                <div className="flex-1 min-w-0">
                                  <span className="font-medium text-slate-700">{node.title}</span>
                                  <span className="text-slate-500 ml-2">
                                    ({node.document}, p.{node.page_ref})
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                    <User size={18} className="text-slate-600" />
                  </div>
                )}
              </div>
            ))
          )}
          
          {isGenerating && (
            <div className="flex gap-4 max-w-4xl mx-auto">
              <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                <Loader2 size={18} className="animate-spin text-indigo-600" />
              </div>
              <div className="px-5 py-3 bg-white text-slate-500 text-sm">
                AI가 규정을 분석하고 있습니다...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {currentSessionId && (
          <div className="bg-white p-4 md:pb-6 border-t border-slate-100">
            {selectedNode && (
              <div className="max-w-3xl mx-auto mb-3 flex items-center gap-2 text-xs bg-indigo-50 px-4 py-2 rounded-lg border border-indigo-200">
                <span className="text-indigo-700">📌 선택된 섹션:</span>
                <span className="font-medium text-indigo-900">{selectedNode.title}</span>
                {selectedNode.page_ref && (
                  <span className="text-indigo-600">(p.{selectedNode.page_ref})</span>
                )}
                <button
                  onClick={() => {
                    setSelectedNode(null);
                    toast.success("섹션 선택 해제됨");
                  }}
                  className="ml-auto text-indigo-600 hover:text-indigo-800"
                  aria-label="섹션 선택 해제"
                >
                  ✕
                </button>
              </div>
            )}
            <div className="max-w-3xl mx-auto relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !isGenerating && handleSendMessage()}
                placeholder={selectedNode ? `"${selectedNode.title}" 섹션에 대해 질문하기...` : "규정에 대해 궁금한 점을 입력하세요..."}
                disabled={isGenerating}
                className="w-full bg-[#f0f4f9] hover:bg-[#e9eef6] focus:bg-white border-2 border-transparent focus:border-indigo-200 rounded-full pl-6 pr-14 py-4 text-slate-700 placeholder:text-slate-400 focus:outline-none transition-all shadow-sm"
                aria-label="질문 입력"
              />
              <button 
                onClick={handleSendMessage}
                disabled={!input.trim() || isGenerating}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
                aria-label="전송"
              >
                <Send size={18} />
              </button>
            </div>
            <div className="text-center mt-2 text-xs text-slate-400">
              AI 답변은 업로드된 문서에 기반하지만, 중요한 결정 시 반드시 원문을 재확인하시기 바랍니다.
            </div>
          </div>
        )}

      </main>

      {showTree && treeData && (
        <aside className="w-96 bg-white border-l border-slate-200 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FolderTree size={18} className="text-indigo-600" />
              <h3 className="font-semibold text-slate-800">문서 구조</h3>
            </div>
            <button
              onClick={() => {
                setShowTree(false);
                setSelectedNode(null);
              }}
              className="p-1 hover:bg-slate-100 rounded"
              aria-label="트리 닫기"
            >
              <PanelLeft size={18} className="text-slate-500" />
            </button>
          </div>
          
          <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
            <div className="text-sm font-medium text-slate-700 mb-1">{treeData.document_name}</div>
            <div className="text-xs text-slate-500">
              💡 팁: <span className="font-medium">Shift + 클릭</span>으로 섹션 선택 후 질문하기
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {renderTreeNode(treeData.tree)}
          </div>
        </aside>
      )}

      {showPdfViewer && pdfFile && (
        <div className="fixed inset-0 bg-black bg-opacity-70 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col">
            <div className="p-4 border-b flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-3">
                <FileText size={20} className="text-indigo-600" />
                <h3 className="font-semibold text-slate-800">{pdfFile}</h3>
                <span className="text-sm text-slate-500">
                  (페이지 {pdfPage})
                </span>
              </div>
              <button
                onClick={() => setShowPdfViewer(false)}
                className="p-2 hover:bg-red-100 text-red-600 rounded-lg transition-colors"
                title="닫기"
              >
                <X size={20} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <iframe
                src={`${API_BASE_URL}/pdf/${pdfFile}#page=${pdfPage}`}
                className="w-full h-full border-0"
                title="PDF Viewer"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}