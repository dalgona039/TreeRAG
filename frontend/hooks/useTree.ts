import { useState } from "react";
import { toast } from "react-hot-toast";
import { api } from "@/lib/api";
import type { TreeData, TreeNode } from "@/lib/types";

export function useTree() {
  const [showTree, setShowTree] = useState(false);
  const [treeData, setTreeData] = useState<TreeData | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);

  const loadTreeStructure = async (indexFilename: string, t: any) => {
    try {
      const data = await api.loadTree(indexFilename);
      setTreeData(data);
      setShowTree(true);
      setExpandedNodes(new Set([data.tree.id]));
      toast.success(`${t.treeLoaded}: ${data.document_name}`);
    } catch (error) {
      toast.error(t.treeLoadFailed);
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

  const handleNodeClick = (node: TreeNode, hasChildren: boolean, e: React.MouseEvent, setInput: (val: string) => void, t: any) => {
    if (hasChildren) {
      toggleNode(node.id);
    }
    
    if (e.shiftKey) {
      e.stopPropagation();
      setSelectedNode(node);
      
      const question = `"${node.title}" 섹션에 대해 자세히 설명해주세요.${node.page_ref ? ` (페이지 ${node.page_ref})` : ''}`;
      setInput(question);
      toast.success(`${t.nodeSelected}: ${node.title}`);
    }
  };

  return {
    showTree,
    setShowTree,
    treeData,
    expandedNodes,
    selectedNode,
    setSelectedNode,
    loadTreeStructure,
    toggleNode,
    handleNodeClick,
  };
}
