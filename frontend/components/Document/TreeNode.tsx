import { ChevronDown, ChevronRight } from "lucide-react";
import type { TreeNode as TreeNodeType } from "@/lib/types";

interface TreeNodeProps {
  node: TreeNodeType;
  level?: number;
  expandedNodes: Set<string>;
  selectedNode: TreeNodeType | null;
  onNodeClick: (node: TreeNodeType, hasChildren: boolean, e: React.MouseEvent) => void;
}

export default function TreeNode({ 
  node, 
  level = 0, 
  expandedNodes, 
  selectedNode,
  onNodeClick 
}: TreeNodeProps) {
  const isExpanded = expandedNodes.has(node.id);
  const hasChildren = node.children && node.children.length > 0;
  const isSelected = selectedNode?.id === node.id;
  
  return (
    <div className="mb-1">
      <div 
        className={`flex items-start gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
          level > 0 ? 'ml-' + (level * 4) : ''
        } ${
          isSelected ? 'bg-indigo-100 border border-indigo-300' : 'hover:bg-slate-50'
        }`}
        onClick={(e) => onNodeClick(node, !!hasChildren, e)}
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
          {node.children!.map(child => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              expandedNodes={expandedNodes}
              selectedNode={selectedNode}
              onNodeClick={onNodeClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}
