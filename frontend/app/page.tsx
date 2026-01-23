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
  Trash2, Copy, Check, ChevronRight, ChevronDown, FolderTree
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

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  comparison?: ComparisonResult;
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
const STORAGE_KEY = "medi-reg-sessions";

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
      const res = await axios.post(`${API_BASE_URL}/chat`, {
        question: userMsg,
        index_filenames: currentSession.indexFiles,
      });
      
      const botMsg = res.data.answer;
      const citations = res.data.citations || [];
      const comparison = res.data.comparison || null;

      setSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { 
              ...session, 
              messages: [...updatedMessages, { 
                role: "assistant", 
                content: botMsg,
                citations,
                comparison 
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
              Medi-Reg Master <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">AI</span>
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
            <button
              onClick={() => loadTreeStructure(currentSession.indexFiles[0])}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm rounded-lg transition-colors"
              title="문서 구조 보기"
            >
              <FolderTree size={16} />
              트리 구조
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth bg-white">
          {!currentSessionId ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-80 pb-20">
              <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-xl">
                <FileText className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-slate-700 mb-2">Medi-Reg Master</h2>
              <p className="max-w-md text-center text-slate-500">
                상단에서 규제 문서를 업로드하면 자동으로 분석합니다.<br/>
                AI가 문서를 구조화하여 법적 근거 기반의 상담을 제공합니다.
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
                        <span 
                          key={i}
                          className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full border border-indigo-100"
                        >
                          📎 {citation}
                        </span>
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
            <div className="max-w-3xl mx-auto relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !isGenerating && handleSendMessage()}
                placeholder="규정에 대해 궁금한 점을 입력하세요..."
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
              AI 답변은 규제 문서에 기반하지만, 반드시 원문을 재확인하시기 바랍니다.
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
              onClick={() => setShowTree(false)}
              className="p-1 hover:bg-slate-100 rounded"
              aria-label="트리 닫기"
            >
              <PanelLeft size={18} className="text-slate-500" />
            </button>
          </div>
          
          <div className="px-4 py-2 bg-slate-50 border-b border-slate-100">
            <div className="text-sm font-medium text-slate-700">{treeData.document_name}</div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {renderTreeNode(treeData.tree)}
          </div>
        </aside>
      )}
    </div>
  );
}