import { useState } from "react";
import { Bot, User, Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Message } from "@/lib/types";

interface MessageItemProps {
  message: Message;
  index: number;
  onCitationClick: (citation: string) => void;
  onCopy: (text: string, id: string) => void;
  copiedId: string | null;
  t: any;
}

export default function MessageItem({ 
  message, 
  index, 
  onCitationClick, 
  onCopy, 
  copiedId,
  t 
}: MessageItemProps) {
  return (
    <div className={`flex gap-4 max-w-4xl mx-auto ${message.role === 'user' ? 'justify-end' : ''}`}>
      {message.role === 'assistant' && (
        <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0 mt-1">
          <Bot size={18} className="text-indigo-600" />
        </div>
      )}

      <div className="flex-1 min-w-0">
        <div 
          className={`group relative px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm ${
            message.role === 'user' 
              ? "bg-[#e7effe] text-slate-800 rounded-br-none ml-auto max-w-[80%]" 
              : "bg-white border border-slate-100 text-slate-800 rounded-tl-none"
          }`}
        >
          {message.role === 'assistant' ? (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ) : (
            <p className="whitespace-pre-wrap">{message.content}</p>
          )}
          
          <button
            onClick={() => onCopy(message.content, `${index}`)}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 p-1.5 bg-slate-100 hover:bg-slate-200 rounded transition-opacity"
            aria-label="복사"
          >
            {copiedId === `${index}` ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>
        
        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2 ml-1">
            {message.citations.map((citation, i) => (
              <button
                key={i}
                onClick={() => onCitationClick(citation)}
                className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full border border-indigo-100 hover:bg-indigo-100 cursor-pointer transition-colors"
              >
                📎 {citation}
              </button>
            ))}
          </div>
        )}
        
        {message.hallucination_warning && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-2">
              <span className="text-lg">⚠️</span>
              <div className="flex-1">
                <h4 className="font-semibold text-red-900 mb-1">Hallucination Warning</h4>
                <p className="text-sm text-red-700">
                  {message.hallucination_warning.message}
                </p>
                <p className="text-xs text-red-600 mt-1">
                  Overall confidence: {(message.hallucination_warning.overall_confidence * 100).toFixed(1)}%
                  (threshold: {(message.hallucination_warning.threshold * 100)}%)
                </p>
                <p className="text-xs text-red-600 mt-1 font-medium">
                  ⚠️ This answer may not be grounded in the documents. Please verify with original sources.
                </p>
              </div>
            </div>
          </div>
        )}
        
        {message.comparison && message.comparison.has_comparison && (
          <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-xl">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
                <span className="text-lg">📊</span>
              </div>
              <h4 className="font-semibold text-amber-900">{t.comparisonAnalysis}</h4>
            </div>
            
            <div className="text-sm text-amber-800 mb-2">
              <strong>{t.comparisonTarget}:</strong> {message.comparison.documents_compared.join(" ↔ ")}
            </div>
            
            {message.comparison.commonalities && (
              <div className="mb-3">
                <div className="font-medium text-green-700 mb-1">✓ {t.commonalities}</div>
                <div className="text-sm text-gray-700 bg-white p-2 rounded">
                  {message.comparison.commonalities}
                </div>
              </div>
            )}
            
            {message.comparison.differences && (
              <div>
                <div className="font-medium text-red-700 mb-1">⚠ {t.differences}</div>
                <div className="text-sm text-gray-700 bg-white p-2 rounded overflow-x-auto">
                  <ReactMarkdown>{message.comparison.differences}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {message.resolved_references && message.resolved_references.length > 0 && (
          <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-xl">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                <span className="text-lg">🔗</span>
              </div>
              <h4 className="font-semibold text-purple-900">{t.crossReferenceResolved}</h4>
            </div>
            <div className="text-xs text-purple-700 mb-2">
              {t.crossReferenceDesc.replace('{count}', message.resolved_references.length.toString())}
            </div>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {message.resolved_references.map((ref, i) => (
                <div key={i} className="bg-white p-2 rounded text-sm">
                  <div className="font-medium text-purple-700">{ref.title}</div>
                  {ref.page_ref && (
                    <div className="text-xs text-slate-500 mt-1">{t.page}: {ref.page_ref}</div>
                  )}
                  {ref.summary && (
                    <div className="text-xs text-slate-600 mt-1 line-clamp-2">{ref.summary}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {message.traversal_info && message.traversal_info.used_deep_traversal && (
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
                <span className="ml-2 text-slate-700">{message.traversal_info.nodes_visited.length}개</span>
              </div>
              <div className="bg-white p-2 rounded">
                <span className="text-blue-600 font-medium">선택된 노드:</span>
                <span className="ml-2 text-slate-700">{message.traversal_info.nodes_selected.length}개</span>
              </div>
              <div className="bg-white p-2 rounded">
                <span className="text-blue-600 font-medium">최대 깊이:</span>
                <span className="ml-2 text-slate-700">{message.traversal_info.max_depth}</span>
              </div>
              <div className="bg-white p-2 rounded">
                <span className="text-blue-600 font-medium">브랜치 수:</span>
                <span className="ml-2 text-slate-700">{message.traversal_info.max_branches}</span>
              </div>
            </div>

            {message.traversal_info.nodes_selected.length > 0 && (
              <div>
                <div className="font-medium text-blue-700 mb-2 text-sm">선택된 섹션:</div>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {message.traversal_info.nodes_selected.map((node, i) => (
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

      {message.role === 'user' && (
        <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
          <User size={18} className="text-slate-600" />
        </div>
      )}
    </div>
  );
}
