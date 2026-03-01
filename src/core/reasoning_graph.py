
import json
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from src.config import Config


class EdgeType(Enum):
    """Types of reasoning relationships between nodes."""
    CAUSE_EFFECT = "cause_effect"      # A causes B
    SUPPORT = "support"                 # A supports/justifies B
    CONTRAST = "contrast"               # A contrasts/contradicts B
    ELABORATION = "elaboration"         # B elaborates on A
    TEMPORAL = "temporal"               # A precedes B temporally
    REFERENCE = "reference"             # A references B
    DEFINITION = "definition"           # A defines concept in B
    EXAMPLE = "example"                 # B is an example of A
    PARENT_CHILD = "parent_child"       # Hierarchical (from tree)
    
    @classmethod
    def from_string(cls, s: str) -> "EdgeType":
        for member in cls:
            if member.value == s.lower():
                return member
        return cls.REFERENCE


@dataclass
class ReasoningEdge:
    """Represents a reasoning relationship between two nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float  # 0.0 to 1.0
    description: str = ""
    bidirectional: bool = False
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))
    
    def __eq__(self, other):
        if not isinstance(other, ReasoningEdge):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id and 
                self.edge_type == other.edge_type)


@dataclass
class GraphNode:
    """Node in the reasoning graph with content and metadata."""
    id: str
    title: str
    summary: str
    page_ref: str = ""
    text: Optional[str] = None
    depth: int = 0
    embedding: Optional[List[float]] = None
    
    def content_hash(self) -> str:
        content = f"{self.title}|{self.summary}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class ReasoningPath:
    """A chain of reasoning from source to target."""
    nodes: List[GraphNode]
    edges: List[ReasoningEdge]
    total_confidence: float
    path_type: str  # e.g., "causal_chain", "supporting_evidence"
    
    def __len__(self):
        return len(self.nodes)
    
    def __lt__(self, other):
        return self.total_confidence > other.total_confidence


class ReasoningGraph:
    """
    Graph structure for semantic reasoning over document sections.
    
    Extends the hierarchical tree with inferred reasoning edges
    to enable multi-hop question answering and evidence chains.
    """
    
    def __init__(self, document_name: str):
        self.document_name = document_name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[ReasoningEdge] = []
        self.adjacency: Dict[str, List[ReasoningEdge]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[ReasoningEdge]] = defaultdict(list)
        self._edge_inference_cache: Dict[str, ReasoningEdge] = {}
    
    def build_from_tree(
        self, 
        tree: Dict[str, Any],
        infer_edges: bool = True,
        max_edge_distance: int = 2
    ) -> "ReasoningGraph":
        """
        Build reasoning graph from tree structure.
        
        Args:
            tree: The document tree structure
            infer_edges: Whether to use LLM to infer reasoning edges
            max_edge_distance: Max tree distance for edge inference
        
        Returns:
            Self for method chaining
        """
        # Step 1: Extract all nodes from tree
        self._extract_nodes_from_tree(tree, depth=0)
        print(f"📊 Extracted {len(self.nodes)} nodes from tree")
        
        # Step 2: Add hierarchical edges (parent-child)
        self._add_hierarchical_edges(tree)
        print(f"🔗 Added {len(self.edges)} hierarchical edges")
        
        # Step 3: Infer semantic edges between siblings and nearby nodes
        if infer_edges:
            sibling_groups = self._get_sibling_groups(tree)
            inferred_count = self._infer_semantic_edges(sibling_groups, max_edge_distance)
            print(f"🧠 Inferred {inferred_count} semantic edges")
        
        return self
    
    def _extract_nodes_from_tree(
        self, 
        node: Dict[str, Any], 
        depth: int
    ) -> None:
        """Recursively extract nodes from tree."""
        node_id = node.get("id", f"node_{len(self.nodes)}")
        
        graph_node = GraphNode(
            id=node_id,
            title=node.get("title", "Untitled"),
            summary=node.get("summary", ""),
            page_ref=node.get("page_ref", ""),
            text=node.get("text"),
            depth=depth
        )
        self.nodes[node_id] = graph_node
        
        children = node.get("children", [])
        if children:
            for child in children:
                self._extract_nodes_from_tree(child, depth + 1)
    
    def _add_hierarchical_edges(self, node: Dict[str, Any]) -> None:
        """Add parent-child edges from tree structure."""
        node_id = node.get("id", "")
        children = node.get("children", [])
        
        if children:
            for child in children:
                child_id = child.get("id", "")
                if node_id and child_id:
                    edge = ReasoningEdge(
                        source_id=node_id,
                        target_id=child_id,
                        edge_type=EdgeType.PARENT_CHILD,
                        confidence=1.0,
                        description="Parent-child hierarchy"
                    )
                    self._add_edge(edge)
                self._add_hierarchical_edges(child)
    
    def _get_sibling_groups(
        self, 
        node: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """Get groups of sibling nodes for edge inference."""
        groups = []
        children = node.get("children", [])
        
        if len(children) > 1:
            groups.append(children)
        
        for child in children:
            groups.extend(self._get_sibling_groups(child))
        
        return groups
    
    def _infer_semantic_edges(
        self, 
        sibling_groups: List[List[Dict[str, Any]]],
        max_distance: int
    ) -> int:
        """Use LLM to infer reasoning edges between sibling nodes."""
        inferred_count = 0
        
        for group in sibling_groups:
            if len(group) < 2:
                continue
            
            # Compare each pair within distance limit
            for i, node_a in enumerate(group):
                for j, node_b in enumerate(group):
                    if i >= j or abs(i - j) > max_distance:
                        continue
                    
                    edge = self._infer_edge_between(node_a, node_b)
                    if edge and edge.confidence >= 0.6:
                        self._add_edge(edge)
                        inferred_count += 1
        
        return inferred_count
    
    def _infer_edge_between(
        self, 
        node_a: Dict[str, Any], 
        node_b: Dict[str, Any]
    ) -> Optional[ReasoningEdge]:
        """Use LLM to infer reasoning relationship between two nodes."""
        id_a = node_a.get("id", "")
        id_b = node_b.get("id", "")
        
        cache_key = f"{id_a}:{id_b}"
        if cache_key in self._edge_inference_cache:
            return self._edge_inference_cache[cache_key]
        
        prompt = f"""Analyze the semantic relationship between two document sections.

Section A:
- Title: {node_a.get('title', '')}
- Summary: {node_a.get('summary', '')[:500]}

Section B:
- Title: {node_b.get('title', '')}
- Summary: {node_b.get('summary', '')[:500]}

Determine if there is a reasoning relationship between these sections.

Possible relationship types:
- cause_effect: A causes or leads to B
- support: A provides evidence/justification for B
- contrast: A contrasts or contradicts B
- elaboration: B elaborates or expands on A
- temporal: A precedes B in time/sequence
- reference: A references concepts in B
- definition: A defines terms used in B
- example: B is an example of concepts in A
- none: No significant relationship

Respond in JSON format:
{{
  "relationship": "<type or 'none'>",
  "confidence": <0.0-1.0>,
  "direction": "a_to_b" or "b_to_a" or "bidirectional",
  "description": "<brief explanation>"
}}

JSON only:"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                return None
            
            result = json.loads(response.text)
            rel_type = result.get("relationship", "none")
            
            if rel_type == "none":
                return None
            
            direction = result.get("direction", "a_to_b")
            source_id = id_a if direction != "b_to_a" else id_b
            target_id = id_b if direction != "b_to_a" else id_a
            
            edge = ReasoningEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=EdgeType.from_string(rel_type),
                confidence=result.get("confidence", 0.5),
                description=result.get("description", ""),
                bidirectional=(direction == "bidirectional")
            )
            
            self._edge_inference_cache[cache_key] = edge
            return edge
            
        except Exception as e:
            print(f"Edge inference error: {e}")
            return None
    
    def _add_edge(self, edge: ReasoningEdge) -> None:
        """Add an edge to the graph, updating adjacency lists."""
        if edge not in self.edges:
            self.edges.append(edge)
            self.adjacency[edge.source_id].append(edge)
            self.reverse_adjacency[edge.target_id].append(edge)
            
            if edge.bidirectional:
                reverse_edge = ReasoningEdge(
                    source_id=edge.target_id,
                    target_id=edge.source_id,
                    edge_type=edge.edge_type,
                    confidence=edge.confidence,
                    description=edge.description,
                    bidirectional=True
                )
                if reverse_edge not in self.edges:
                    self.edges.append(reverse_edge)
                    self.adjacency[reverse_edge.source_id].append(reverse_edge)
                    self.reverse_adjacency[reverse_edge.target_id].append(reverse_edge)
    
    def find_reasoning_paths(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        min_confidence: float = 0.5
    ) -> List[ReasoningPath]:
        """
        Find reasoning paths relevant to the query.
        
        Uses a confidence-weighted graph search to find the most
        relevant chains of reasoning for multi-hop questions.
        
        Args:
            query: The user's question
            max_hops: Maximum path length
            top_k: Number of paths to return
            min_confidence: Minimum path confidence threshold
        
        Returns:
            List of reasoning paths sorted by confidence
        """
        # Step 1: Find seed nodes relevant to query
        seed_nodes = self._find_seed_nodes(query, top_k=max(5, top_k))
        
        if not seed_nodes:
            return []
        
        # Step 2: Expand from seeds using BFS with confidence
        all_paths: List[ReasoningPath] = []
        
        for seed_id, seed_score in seed_nodes:
            paths = self._expand_from_node(
                seed_id, 
                initial_confidence=seed_score,
                max_hops=max_hops,
                min_confidence=min_confidence
            )
            all_paths.extend(paths)
        
        # Step 3: Rank and deduplicate paths
        all_paths.sort(key=lambda p: p.total_confidence, reverse=True)
        
        seen_node_sets: Set[frozenset] = set()
        unique_paths = []
        for path in all_paths:
            node_set = frozenset(n.id for n in path.nodes)
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                unique_paths.append(path)
                if len(unique_paths) >= top_k:
                    break
        
        return unique_paths
    
    def _find_seed_nodes(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find nodes most relevant to the query as starting points."""
        if not self.nodes:
            return []
        
        prompt = f"""Given a query, identify the most relevant document sections.

Query: {query}

Available sections:
{self._format_nodes_for_prompt(list(self.nodes.values())[:30])}

Return the IDs of the top {top_k} most relevant sections with relevance scores.

JSON format:
{{
  "relevant_sections": [
    {{"id": "<node_id>", "score": <0.0-1.0>, "reason": "<brief reason>"}}
  ]
}}

JSON only:"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                return []
            
            result = json.loads(response.text)
            sections = result.get("relevant_sections", [])
            
            seeds = []
            for s in sections[:top_k]:
                node_id = s.get("id", "")
                if node_id in self.nodes:
                    seeds.append((node_id, s.get("score", 0.5)))
            
            return seeds
            
        except Exception as e:
            print(f"Seed node finding error: {e}")
            # Fallback: return first few nodes
            return [(node_id, 0.5) for node_id in list(self.nodes.keys())[:top_k]]
    
    def _format_nodes_for_prompt(self, nodes: List[GraphNode]) -> str:
        """Format nodes for LLM prompt."""
        lines = []
        for node in nodes:
            lines.append(f"- [{node.id}] {node.title}: {node.summary[:100]}...")
        return "\n".join(lines)
    
    def _expand_from_node(
        self,
        start_id: str,
        initial_confidence: float,
        max_hops: int,
        min_confidence: float
    ) -> List[ReasoningPath]:
        """Expand reasoning paths from a starting node using priority queue."""
        if start_id not in self.nodes:
            return []
        
        start_node = self.nodes[start_id]
        initial_path = ReasoningPath(
            nodes=[start_node],
            edges=[],
            total_confidence=initial_confidence,
            path_type="seed"
        )
        
        # Priority queue: (-confidence, path)
        pq: List[Tuple[float, int, ReasoningPath]] = [
            (-initial_confidence, 0, initial_path)
        ]
        
        completed_paths: List[ReasoningPath] = []
        visited_states: Set[Tuple[str, int]] = set()
        counter = 1
        
        while pq and len(completed_paths) < 20:
            neg_conf, _, current_path = heapq.heappop(pq)
            current_conf = -neg_conf
            
            if current_conf < min_confidence:
                continue
            
            current_node = current_path.nodes[-1]
            state = (current_node.id, len(current_path.nodes))
            
            if state in visited_states:
                continue
            visited_states.add(state)
            
            if len(current_path.nodes) > 1:
                completed_paths.append(current_path)
            
            if len(current_path.nodes) >= max_hops:
                continue
            
            # Expand to neighbors
            for edge in self.adjacency[current_node.id]:
                if edge.target_id not in self.nodes:
                    continue
                
                target_node = self.nodes[edge.target_id]
                
                # Skip if already in path (no cycles)
                if any(n.id == target_node.id for n in current_path.nodes):
                    continue
                
                # Calculate new path confidence
                new_confidence = current_conf * edge.confidence
                
                if new_confidence < min_confidence:
                    continue
                
                new_path = ReasoningPath(
                    nodes=current_path.nodes + [target_node],
                    edges=current_path.edges + [edge],
                    total_confidence=new_confidence,
                    path_type=self._classify_path_type(current_path.edges + [edge])
                )
                
                heapq.heappush(pq, (-new_confidence, counter, new_path))
                counter += 1
        
        return completed_paths
    
    def _classify_path_type(self, edges: List[ReasoningEdge]) -> str:
        """Classify the type of reasoning path based on edges."""
        if not edges:
            return "single_node"
        
        edge_types = [e.edge_type for e in edges]
        
        if all(t == EdgeType.CAUSE_EFFECT for t in edge_types):
            return "causal_chain"
        elif all(t == EdgeType.SUPPORT for t in edge_types):
            return "supporting_evidence"
        elif any(t == EdgeType.CONTRAST for t in edge_types):
            return "comparative_analysis"
        elif all(t == EdgeType.TEMPORAL for t in edge_types):
            return "temporal_sequence"
        elif all(t == EdgeType.PARENT_CHILD for t in edge_types):
            return "hierarchical"
        else:
            return "mixed_reasoning"
    
    def get_node_context(
        self, 
        node_id: str, 
        include_neighbors: bool = True
    ) -> Dict[str, Any]:
        """Get context for a node including its reasoning connections."""
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        context = {
            "node": {
                "id": node.id,
                "title": node.title,
                "summary": node.summary,
                "page_ref": node.page_ref,
                "depth": node.depth
            },
            "outgoing_edges": [],
            "incoming_edges": []
        }
        
        if include_neighbors:
            for edge in self.adjacency[node_id]:
                if edge.target_id in self.nodes:
                    target = self.nodes[edge.target_id]
                    context["outgoing_edges"].append({
                        "target_id": target.id,
                        "target_title": target.title,
                        "edge_type": edge.edge_type.value,
                        "confidence": edge.confidence,
                        "description": edge.description
                    })
            
            for edge in self.reverse_adjacency[node_id]:
                if edge.source_id in self.nodes:
                    source = self.nodes[edge.source_id]
                    context["incoming_edges"].append({
                        "source_id": source.id,
                        "source_title": source.title,
                        "edge_type": edge.edge_type.value,
                        "confidence": edge.confidence,
                        "description": edge.description
                    })
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary for storage."""
        return {
            "document_name": self.document_name,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "title": node.title,
                    "summary": node.summary,
                    "page_ref": node.page_ref,
                    "depth": node.depth
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "confidence": edge.confidence,
                    "description": edge.description,
                    "bidirectional": edge.bidirectional
                }
                for edge in self.edges
            ],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "edge_types": self._count_edge_types()
            }
        }
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type."""
        counts: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            counts[edge.edge_type.value] += 1
        return dict(counts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningGraph":
        """Deserialize graph from dictionary."""
        graph = cls(data.get("document_name", "unknown"))
        
        for node_id, node_data in data.get("nodes", {}).items():
            graph.nodes[node_id] = GraphNode(
                id=node_data["id"],
                title=node_data["title"],
                summary=node_data.get("summary", ""),
                page_ref=node_data.get("page_ref", ""),
                depth=node_data.get("depth", 0)
            )
        
        for edge_data in data.get("edges", []):
            edge = ReasoningEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=EdgeType.from_string(edge_data["edge_type"]),
                confidence=edge_data.get("confidence", 1.0),
                description=edge_data.get("description", ""),
                bidirectional=edge_data.get("bidirectional", False)
            )
            graph._add_edge(edge)
        
        return graph


class GraphNavigator:
    """
    High-level navigator for reasoning-based document search.
    
    Combines tree traversal with graph reasoning for enhanced
    multi-hop question answering.
    """
    
    def __init__(self, graph: ReasoningGraph):
        self.graph = graph
    
    def search_with_reasoning(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        include_paths: bool = True
    ) -> Dict[str, Any]:
        """
        Search for relevant information with reasoning paths.
        
        Args:
            query: User's question
            max_hops: Maximum reasoning chain length
            top_k: Number of results to return
            include_paths: Whether to include full reasoning paths
        
        Returns:
            Search results with reasoning context
        """
        # Find reasoning paths
        paths = self.graph.find_reasoning_paths(
            query=query,
            max_hops=max_hops,
            top_k=top_k
        )
        
        # Collect unique nodes from paths
        seen_nodes: Set[str] = set()
        result_nodes = []
        
        for path in paths:
            for node in path.nodes:
                if node.id not in seen_nodes:
                    seen_nodes.add(node.id)
                    result_nodes.append({
                        "id": node.id,
                        "title": node.title,
                        "summary": node.summary,
                        "page_ref": node.page_ref,
                        "depth": node.depth,
                        "context": self.graph.get_node_context(node.id)
                    })
        
        result = {
            "query": query,
            "nodes_found": len(result_nodes),
            "nodes": result_nodes[:top_k * 2],
            "reasoning_stats": {
                "paths_found": len(paths),
                "max_path_confidence": max((p.total_confidence for p in paths), default=0),
                "path_types": list(set(p.path_type for p in paths))
            }
        }
        
        if include_paths:
            result["reasoning_paths"] = [
                {
                    "nodes": [n.title for n in path.nodes],
                    "edges": [
                        {
                            "from": e.source_id,
                            "to": e.target_id,
                            "type": e.edge_type.value,
                            "confidence": e.confidence
                        }
                        for e in path.edges
                    ],
                    "total_confidence": path.total_confidence,
                    "path_type": path.path_type
                }
                for path in paths[:top_k]
            ]
        
        return result
    
    def explain_connection(
        self,
        node_a_id: str,
        node_b_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Explain the reasoning connection between two nodes.
        
        Uses shortest path + edge descriptions to generate
        an explanation of how two concepts are related.
        """
        if node_a_id not in self.graph.nodes or node_b_id not in self.graph.nodes:
            return None
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(node_a_id, [node_a_id], [])])
        visited = {node_a_id}
        
        while queue:
            current_id, node_path, edge_path = queue.popleft()
            
            if current_id == node_b_id:
                # Found path
                return {
                    "connected": True,
                    "path_length": len(node_path) - 1,
                    "nodes": [
                        self.graph.nodes[nid].title 
                        for nid in node_path
                    ],
                    "edges": [
                        {
                            "type": e.edge_type.value,
                            "description": e.description,
                            "confidence": e.confidence
                        }
                        for e in edge_path
                    ],
                    "explanation": self._generate_path_explanation(node_path, edge_path)
                }
            
            for edge in self.graph.adjacency[current_id]:
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((
                        edge.target_id,
                        node_path + [edge.target_id],
                        edge_path + [edge]
                    ))
        
        return {
            "connected": False,
            "path_length": -1,
            "explanation": "No reasoning path found between these sections."
        }
    
    def _generate_path_explanation(
        self,
        node_path: List[str],
        edge_path: List[ReasoningEdge]
    ) -> str:
        """Generate natural language explanation of reasoning path."""
        if not edge_path:
            return "Direct reference."
        
        explanations = []
        for i, edge in enumerate(edge_path):
            source = self.graph.nodes[node_path[i]]
            target = self.graph.nodes[node_path[i + 1]]
            
            type_phrases = {
                EdgeType.CAUSE_EFFECT: f"'{source.title}' leads to '{target.title}'",
                EdgeType.SUPPORT: f"'{source.title}' supports '{target.title}'",
                EdgeType.CONTRAST: f"'{source.title}' contrasts with '{target.title}'",
                EdgeType.ELABORATION: f"'{target.title}' elaborates on '{source.title}'",
                EdgeType.TEMPORAL: f"'{source.title}' precedes '{target.title}'",
                EdgeType.REFERENCE: f"'{source.title}' references '{target.title}'",
                EdgeType.DEFINITION: f"'{source.title}' defines concepts in '{target.title}'",
                EdgeType.EXAMPLE: f"'{target.title}' exemplifies '{source.title}'",
                EdgeType.PARENT_CHILD: f"'{target.title}' is a section of '{source.title}'"
            }
            
            phrase = type_phrases.get(
                edge.edge_type, 
                f"'{source.title}' relates to '{target.title}'"
            )
            explanations.append(phrase)
        
        return " → ".join(explanations)
