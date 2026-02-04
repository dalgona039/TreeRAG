
"use client";

import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import type { Components } from 'react-markdown';

interface SafeMarkdownProps {
  content: string;
  className?: string;
}

export default function SafeMarkdown({ content, className = '' }: SafeMarkdownProps) {
  const components: Components = {
    a: ({ node, ...props }) => (
      <a
        {...props}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 hover:underline"
      />
    ),
    
    code: ({ node, inline, ...props }) => 
      inline ? (
        <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props} />
      ) : (
        <code className="block bg-gray-100 p-3 rounded text-sm font-mono overflow-x-auto" {...props} />
      ),
    
    blockquote: ({ node, ...props }) => (
      <blockquote className="border-l-4 border-gray-300 pl-4 italic my-4" {...props} />
    ),
    
    ul: ({ node, ...props }) => (
      <ul className="list-disc list-inside my-2 space-y-1" {...props} />
    ),
    ol: ({ node, ...props }) => (
      <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
    ),
    
    h1: ({ node, ...props }) => (
      <h1 className="text-2xl font-bold my-4" {...props} />
    ),
    h2: ({ node, ...props }) => (
      <h2 className="text-xl font-bold my-3" {...props} />
    ),
    h3: ({ node, ...props }) => (
      <h3 className="text-lg font-bold my-2" {...props} />
    ),
    
    // 단락
    p: ({ node, ...props }) => (
      <p className="my-2 leading-relaxed" {...props} />
    ),
    
    // 구분선
    hr: ({ node, ...props }) => (
      <hr className="my-4 border-gray-300" {...props} />
    ),
  };

  return (
    <div className={`prose prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        components={components}
        rehypePlugins={[rehypeSanitize]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export function SafeText({ content, className = '' }: SafeMarkdownProps) {
  return (
    <div className={`whitespace-pre-wrap ${className}`}>
      {content}
    </div>
  );
}

export function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
    '/': '&#x2F;',
  };
  
  return text.replace(/[&<>"'/]/g, (char) => map[char] || char);
}

export function isSafeUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const safeProtocols = ['http:', 'https:', 'mailto:'];
    return safeProtocols.includes(parsed.protocol);
  } catch {
    return false;
  }
}
