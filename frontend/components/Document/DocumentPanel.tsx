import { FolderTree, PanelLeft } from "lucide-react";
import TreeNode from "../Document/TreeNode";
import type { TreeData, TreeNode as TreeNodeType } from "@/lib/types";

interface DocumentPanelProps {
  showTree: boolean;
  treeData: TreeData | null;
  expandedNodes: Set<string>;
  selectedNode: TreeNodeType | null;
  onClose: () => void;
  onNodeClick: (node: TreeNodeType, hasChildren: boolean, e: React.MouseEvent) => void;
  t: any;
}

export default function DocumentPanel({ 
  showTree, 
  treeData, 
  expandedNodes, 
  selectedNode,
  onClose, 
  onNodeClick,
  t 
}: DocumentPanelProps) {
  if (!showTree || !treeData) return null;

  return (
    <aside className="w-96 bg-white border-l border-slate-200 flex flex-col overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FolderTree size={18} className="text-indigo-600" />
          <h3 className="font-semibold text-slate-800">{t.treeStructure}</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-100 rounded"
          aria-label={t.closeTree}
        >
          <PanelLeft size={18} className="text-slate-500" />
        </button>
      </div>
      
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
        <div className="text-sm font-medium text-slate-700 mb-1">{treeData.document_name}</div>
        <div className="text-xs text-slate-500">
          💡 {t.tipTreeClick}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <TreeNode
          node={treeData.tree}
          expandedNodes={expandedNodes}
          selectedNode={selectedNode}
          onNodeClick={onNodeClick}
        />
      </div>
    </aside>
  );
}
