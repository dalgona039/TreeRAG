"""
Cross-reference resolver for TreeRAG.
Automatically detects and resolves references like "Section 5.2", "Chapter 3", "표 2" etc.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class ReferenceResolver:
    """Detects and resolves cross-references in user queries."""
    
    # 다양한 참조 패턴 (한글/영문)
    REFERENCE_PATTERNS = [
        # 섹션 참조
        r'(?:Section|섹션|section)\s*(\d+(?:\.\d+)*)',
        r'(\d+(?:\.\d+)+)\s*(?:Section|섹션|section)',
        
        # 장/챕터 참조
        r'(?:Chapter|장|chapter|챕터)\s*(\d+)',
        r'(\d+)\s*(?:장|챕터)',
        
        # 표 참조
        r'(?:Table|표|table)\s*(\d+(?:\.\d+)*)',
        r'표\s*<?\s*(\d+(?:\.\d+)*)\s*>?',
        
        # 그림/도표 참조
        r'(?:Figure|그림|figure|Fig\.|도)\s*(\d+(?:\.\d+)*)',
        r'그림\s*<?\s*(\d+(?:\.\d+)*)\s*>?',
        
        # 부록 참조
        r'(?:Appendix|부록|appendix)\s*([A-Z]|\d+)',
        r'부록\s*([A-Z가-힣]|\d+)',
    ]
    
    def __init__(self, tree_data: Dict[str, Any]):
        """
        Initialize resolver with document tree.
        
        Args:
            tree_data: PageIndex tree structure
        """
        self.tree_data = tree_data
        self.node_index = self._build_node_index()
    
    def _build_node_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Build searchable index of all nodes.
        Returns dict mapping various keys to nodes.
        """
        index = {}
        
        def traverse(node: Dict[str, Any], path: str = ""):
            node_id = node.get("id", "")
            title = node.get("title", "")
            
            # Index by ID
            if node_id:
                index[node_id.lower()] = node
            
            # Index by title
            if title:
                index[title.lower()] = node
                
                # Extract section numbers from title
                section_match = re.search(r'(\d+(?:\.\d+)+)', title)
                if section_match:
                    section_num = section_match.group(1)
                    index[f"section_{section_num}"] = node
                    index[section_num] = node
                
                # Extract chapter numbers
                chapter_match = re.search(r'(?:Chapter|장|챕터)\s*(\d+)', title, re.IGNORECASE)
                if chapter_match:
                    chapter_num = chapter_match.group(1)
                    index[f"chapter_{chapter_num}"] = node
                    index[f"장_{chapter_num}"] = node
                
                # Extract table/figure numbers
                table_match = re.search(r'(?:Table|표)\s*(\d+(?:\.\d+)*)', title, re.IGNORECASE)
                if table_match:
                    table_num = table_match.group(1)
                    index[f"table_{table_num}"] = node
                    index[f"표_{table_num}"] = node
                
                figure_match = re.search(r'(?:Figure|Fig\.|그림|도)\s*(\d+(?:\.\d+)*)', title, re.IGNORECASE)
                if figure_match:
                    figure_num = figure_match.group(1)
                    index[f"figure_{figure_num}"] = node
                    index[f"그림_{figure_num}"] = node
            
            # Recursively index children
            for child in node.get("children", []):
                traverse(child, f"{path}/{title}" if path else title)
        
        if "tree" in self.tree_data:
            traverse(self.tree_data["tree"])
        
        return index
    
    def detect_references(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect all cross-references in text.
        
        Args:
            text: User query or response text
            
        Returns:
            List of (reference_text, reference_key) tuples
        """
        references = []
        
        for pattern in self.REFERENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0)
                ref_number = match.group(1)
                
                # Determine reference type
                if any(keyword in ref_text.lower() for keyword in ['section', '섹션']):
                    ref_key = f"section_{ref_number}"
                elif any(keyword in ref_text.lower() for keyword in ['chapter', '장', '챕터']):
                    ref_key = f"chapter_{ref_number}"
                elif any(keyword in ref_text.lower() for keyword in ['table', '표']):
                    ref_key = f"table_{ref_number}"
                elif any(keyword in ref_text.lower() for keyword in ['figure', 'fig', '그림', '도']):
                    ref_key = f"figure_{ref_number}"
                elif any(keyword in ref_text.lower() for keyword in ['appendix', '부록']):
                    ref_key = f"appendix_{ref_number}"
                else:
                    # Fallback: just use the number
                    ref_key = ref_number
                
                references.append((ref_text, ref_key))
        
        return references
    
    def resolve_reference(self, reference_key: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a reference key to actual node.
        
        Args:
            reference_key: Key like "section_5.2" or "chapter_3"
            
        Returns:
            Node dict or None if not found
        """
        # Try exact match first
        if reference_key.lower() in self.node_index:
            return self.node_index[reference_key.lower()]
        
        # Try without prefix
        if '_' in reference_key:
            suffix = reference_key.split('_', 1)[1]
            if suffix in self.node_index:
                return self.node_index[suffix]
        
        return None
    
    def resolve_all_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect and resolve all references in text.
        
        Args:
            text: User query
            
        Returns:
            List of resolved node dictionaries
        """
        references = self.detect_references(text)
        resolved_nodes = []
        
        for ref_text, ref_key in references:
            node = self.resolve_reference(ref_key)
            if node and node not in resolved_nodes:
                resolved_nodes.append(node)
        
        return resolved_nodes
    
    def format_resolved_context(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Format resolved nodes into context string.
        
        Args:
            nodes: List of resolved nodes
            
        Returns:
            Formatted context string
        """
        if not nodes:
            return ""
        
        context = "\n\n### 📎 Referenced Sections:\n\n"
        
        for i, node in enumerate(nodes, 1):
            title = node.get("title", "Unknown")
            summary = node.get("summary", "")
            page_ref = node.get("page_ref", "")
            
            context += f"**{i}. {title}**"
            if page_ref:
                context += f" ({page_ref})"
            context += "\n"
            
            if summary:
                context += f"{summary}\n\n"
        
        return context
