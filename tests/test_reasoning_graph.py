"""
Tests for Reasoning Graph module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.reasoning_graph import (
    EdgeType,
    ReasoningEdge,
    GraphNode,
    ReasoningPath,
    ReasoningGraph,
    GraphNavigator
)


class TestEdgeType:
    """Tests for EdgeType enum."""
    
    def test_from_string_valid(self):
        assert EdgeType.from_string("cause_effect") == EdgeType.CAUSE_EFFECT
        assert EdgeType.from_string("support") == EdgeType.SUPPORT
        assert EdgeType.from_string("contrast") == EdgeType.CONTRAST
        assert EdgeType.from_string("elaboration") == EdgeType.ELABORATION
        assert EdgeType.from_string("temporal") == EdgeType.TEMPORAL
    
    def test_from_string_invalid(self):
        assert EdgeType.from_string("unknown_type") == EdgeType.REFERENCE
        assert EdgeType.from_string("") == EdgeType.REFERENCE
    
    def test_from_string_case_insensitive(self):
        assert EdgeType.from_string("CAUSE_EFFECT") == EdgeType.CAUSE_EFFECT
        assert EdgeType.from_string("Support") == EdgeType.SUPPORT


class TestReasoningEdge:
    """Tests for ReasoningEdge dataclass."""
    
    def test_edge_creation(self):
        edge = ReasoningEdge(
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.CAUSE_EFFECT,
            confidence=0.9,
            description="A causes B"
        )
        assert edge.source_id == "node_1"
        assert edge.target_id == "node_2"
        assert edge.confidence == 0.9
        assert not edge.bidirectional
    
    def test_edge_hash(self):
        edge1 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.8)
        edge2 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9)
        edge3 = ReasoningEdge("a", "b", EdgeType.CONTRAST, 0.8)
        
        # Same source, target, type → same hash
        assert hash(edge1) == hash(edge2)
        # Different type → different hash
        assert hash(edge1) != hash(edge3)
    
    def test_edge_equality(self):
        edge1 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.8)
        edge2 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9)
        edge3 = ReasoningEdge("a", "c", EdgeType.SUPPORT, 0.8)
        
        assert edge1 == edge2  # Same source, target, type
        assert edge1 != edge3  # Different target


class TestGraphNode:
    """Tests for GraphNode dataclass."""
    
    def test_node_creation(self):
        node = GraphNode(
            id="node_1",
            title="Introduction",
            summary="This is the intro",
            page_ref="1-3",
            depth=0
        )
        assert node.id == "node_1"
        assert node.title == "Introduction"
        assert node.depth == 0
    
    def test_content_hash(self):
        node1 = GraphNode("a", "Title", "Summary")
        node2 = GraphNode("b", "Title", "Summary")
        node3 = GraphNode("a", "Title", "Different")
        
        # Same title+summary → same hash
        assert node1.content_hash() == node2.content_hash()
        # Different summary → different hash
        assert node1.content_hash() != node3.content_hash()


class TestReasoningPath:
    """Tests for ReasoningPath dataclass."""
    
    def test_path_length(self):
        nodes = [
            GraphNode("1", "A", ""),
            GraphNode("2", "B", ""),
            GraphNode("3", "C", "")
        ]
        edges = [
            ReasoningEdge("1", "2", EdgeType.SUPPORT, 0.9),
            ReasoningEdge("2", "3", EdgeType.ELABORATION, 0.8)
        ]
        path = ReasoningPath(
            nodes=nodes,
            edges=edges,
            total_confidence=0.72,
            path_type="supporting_evidence"
        )
        assert len(path) == 3
    
    def test_path_comparison(self):
        path1 = ReasoningPath([], [], 0.9, "a")
        path2 = ReasoningPath([], [], 0.8, "b")
        
        # Higher confidence = lower in comparison (for min-heap)
        assert path1 < path2  # 0.9 > 0.8


class TestReasoningGraph:
    """Tests for ReasoningGraph class."""
    
    @pytest.fixture
    def sample_tree(self):
        return {
            "id": "root",
            "title": "Document",
            "summary": "Main document",
            "children": [
                {
                    "id": "section_1",
                    "title": "Introduction",
                    "summary": "Background and motivation",
                    "children": [
                        {
                            "id": "section_1_1",
                            "title": "Problem Statement",
                            "summary": "The problem we solve"
                        },
                        {
                            "id": "section_1_2",
                            "title": "Contributions",
                            "summary": "Our contributions"
                        }
                    ]
                },
                {
                    "id": "section_2",
                    "title": "Methods",
                    "summary": "Proposed methods",
                    "children": [
                        {
                            "id": "section_2_1",
                            "title": "Algorithm",
                            "summary": "Main algorithm"
                        }
                    ]
                },
                {
                    "id": "section_3",
                    "title": "Results",
                    "summary": "Experimental results"
                }
            ]
        }
    
    @pytest.fixture
    def graph(self):
        return ReasoningGraph("test_document")
    
    def test_build_from_tree_extracts_nodes(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        assert len(graph.nodes) == 7  # root + 3 sections + 3 subsections
        assert "root" in graph.nodes
        assert "section_1" in graph.nodes
        assert "section_2_1" in graph.nodes
    
    def test_build_from_tree_creates_hierarchical_edges(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        # Count parent-child edges
        hierarchical_edges = [
            e for e in graph.edges 
            if e.edge_type == EdgeType.PARENT_CHILD
        ]
        assert len(hierarchical_edges) == 6  # 6 parent-child relationships
    
    def test_adjacency_lists_populated(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        # Root should have 3 outgoing edges
        assert len(graph.adjacency["root"]) == 3
        
        # Section 1 should have 2 outgoing edges (to its children)
        assert len(graph.adjacency["section_1"]) == 2
        
        # Section 3 (leaf) should have no outgoing edges
        assert len(graph.adjacency["section_3"]) == 0
    
    def test_reverse_adjacency_lists(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        # Section 1 should have 1 incoming edge (from root)
        assert len(graph.reverse_adjacency["section_1"]) == 1
        
        # Section 1_1 should have 1 incoming edge (from section_1)
        assert len(graph.reverse_adjacency["section_1_1"]) == 1
    
    def test_add_edge_prevents_duplicates(self, graph):
        edge1 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9)
        edge2 = ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.8)  # Same type
        
        graph._add_edge(edge1)
        graph._add_edge(edge2)
        
        # Should only have one edge
        assert len(graph.edges) == 1
    
    def test_bidirectional_edge(self, graph):
        edge = ReasoningEdge(
            "a", "b", EdgeType.CONTRAST, 0.9, 
            bidirectional=True
        )
        graph.nodes["a"] = GraphNode("a", "A", "")
        graph.nodes["b"] = GraphNode("b", "B", "")
        
        graph._add_edge(edge)
        
        # Should have both directions
        assert len(graph.edges) == 2
        assert any(e.source_id == "a" and e.target_id == "b" for e in graph.edges)
        assert any(e.source_id == "b" and e.target_id == "a" for e in graph.edges)
    
    def test_get_node_context(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        context = graph.get_node_context("section_1")
        
        assert context["node"]["id"] == "section_1"
        assert context["node"]["title"] == "Introduction"
        assert len(context["outgoing_edges"]) == 2  # To children
        assert len(context["incoming_edges"]) == 1  # From root
    
    def test_to_dict_and_from_dict(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        # Serialize
        data = graph.to_dict()
        
        assert data["document_name"] == "test_document"
        assert data["stats"]["node_count"] == 7
        assert data["stats"]["edge_count"] == 6
        
        # Deserialize
        restored = ReasoningGraph.from_dict(data)
        
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)
    
    def test_count_edge_types(self, graph, sample_tree):
        graph.build_from_tree(sample_tree, infer_edges=False)
        
        counts = graph._count_edge_types()
        
        assert counts.get("parent_child", 0) == 6
    
    def test_classify_path_type_causal(self, graph):
        edges = [
            ReasoningEdge("a", "b", EdgeType.CAUSE_EFFECT, 0.9),
            ReasoningEdge("b", "c", EdgeType.CAUSE_EFFECT, 0.8)
        ]
        assert graph._classify_path_type(edges) == "causal_chain"
    
    def test_classify_path_type_supporting(self, graph):
        edges = [
            ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9),
            ReasoningEdge("b", "c", EdgeType.SUPPORT, 0.8)
        ]
        assert graph._classify_path_type(edges) == "supporting_evidence"
    
    def test_classify_path_type_comparative(self, graph):
        edges = [
            ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9),
            ReasoningEdge("b", "c", EdgeType.CONTRAST, 0.8)
        ]
        assert graph._classify_path_type(edges) == "comparative_analysis"
    
    def test_classify_path_type_mixed(self, graph):
        edges = [
            ReasoningEdge("a", "b", EdgeType.SUPPORT, 0.9),
            ReasoningEdge("b", "c", EdgeType.ELABORATION, 0.8)
        ]
        assert graph._classify_path_type(edges) == "mixed_reasoning"


class TestReasoningGraphWithMockedLLM:
    """Tests for reasoning graph with mocked LLM calls."""
    
    @pytest.fixture
    def sample_tree(self):
        return {
            "id": "root",
            "title": "Document",
            "summary": "Main document",
            "children": [
                {"id": "s1", "title": "Cause", "summary": "The cause"},
                {"id": "s2", "title": "Effect", "summary": "The effect"}
            ]
        }
    
    @patch("src.core.reasoning_graph.Config")
    def test_infer_edge_between(self, mock_config, sample_tree):
        mock_response = Mock()
        mock_response.text = '{"relationship": "cause_effect", "confidence": 0.85, "direction": "a_to_b", "description": "Cause leads to effect"}'
        mock_config.CLIENT.models.generate_content.return_value = mock_response
        
        graph = ReasoningGraph("test")
        
        edge = graph._infer_edge_between(
            {"id": "s1", "title": "Cause", "summary": "The cause"},
            {"id": "s2", "title": "Effect", "summary": "The effect"}
        )
        
        assert edge is not None
        assert edge.source_id == "s1"
        assert edge.target_id == "s2"
        assert edge.edge_type == EdgeType.CAUSE_EFFECT
        assert edge.confidence == 0.85
    
    @patch("src.core.reasoning_graph.Config")
    def test_infer_edge_no_relationship(self, mock_config):
        mock_response = Mock()
        mock_response.text = '{"relationship": "none", "confidence": 0.3}'
        mock_config.CLIENT.models.generate_content.return_value = mock_response
        
        graph = ReasoningGraph("test")
        
        edge = graph._infer_edge_between(
            {"id": "a", "title": "A", "summary": "Unrelated A"},
            {"id": "b", "title": "B", "summary": "Unrelated B"}
        )
        
        assert edge is None
    
    @patch("src.core.reasoning_graph.Config")
    def test_find_seed_nodes(self, mock_config):
        mock_response = Mock()
        mock_response.text = '{"relevant_sections": [{"id": "s1", "score": 0.9, "reason": "Highly relevant"}, {"id": "s2", "score": 0.7, "reason": "Somewhat relevant"}]}'
        mock_config.CLIENT.models.generate_content.return_value = mock_response
        
        graph = ReasoningGraph("test")
        graph.nodes["s1"] = GraphNode("s1", "Section 1", "Summary 1")
        graph.nodes["s2"] = GraphNode("s2", "Section 2", "Summary 2")
        
        seeds = graph._find_seed_nodes("test query", top_k=2)
        
        assert len(seeds) == 2
        assert seeds[0] == ("s1", 0.9)
        assert seeds[1] == ("s2", 0.7)


class TestGraphNavigator:
    """Tests for GraphNavigator class."""
    
    @pytest.fixture
    def graph_with_paths(self):
        graph = ReasoningGraph("test")
        
        # Create nodes
        nodes = [
            GraphNode("n1", "Problem", "The problem statement"),
            GraphNode("n2", "Cause", "Root cause analysis"),
            GraphNode("n3", "Solution", "Proposed solution"),
            GraphNode("n4", "Result", "Final results")
        ]
        for node in nodes:
            graph.nodes[node.id] = node
        
        # Create edges forming a causal chain
        edges = [
            ReasoningEdge("n1", "n2", EdgeType.CAUSE_EFFECT, 0.9),
            ReasoningEdge("n2", "n3", EdgeType.SUPPORT, 0.85),
            ReasoningEdge("n3", "n4", EdgeType.CAUSE_EFFECT, 0.8)
        ]
        for edge in edges:
            graph._add_edge(edge)
        
        return graph
    
    def test_explain_connection_found(self, graph_with_paths):
        navigator = GraphNavigator(graph_with_paths)
        
        result = navigator.explain_connection("n1", "n3")
        
        assert result["connected"] is True
        assert result["path_length"] == 2
        assert len(result["nodes"]) == 3
        assert "explanation" in result
    
    def test_explain_connection_not_found(self, graph_with_paths):
        # Add isolated node
        graph_with_paths.nodes["isolated"] = GraphNode("isolated", "Isolated", "Unconnected")
        
        navigator = GraphNavigator(graph_with_paths)
        
        result = navigator.explain_connection("n1", "isolated")
        
        assert result["connected"] is False
        assert result["path_length"] == -1
    
    def test_explain_connection_invalid_nodes(self, graph_with_paths):
        navigator = GraphNavigator(graph_with_paths)
        
        result = navigator.explain_connection("nonexistent", "n1")
        
        assert result is None
    
    def test_search_with_reasoning_structure(self, graph_with_paths):
        navigator = GraphNavigator(graph_with_paths)
        
        # Mock the find_reasoning_paths to avoid LLM calls
        with patch.object(graph_with_paths, 'find_reasoning_paths') as mock_find:
            mock_find.return_value = [
                ReasoningPath(
                    nodes=[graph_with_paths.nodes["n1"], graph_with_paths.nodes["n2"]],
                    edges=[ReasoningEdge("n1", "n2", EdgeType.CAUSE_EFFECT, 0.9)],
                    total_confidence=0.9,
                    path_type="causal_chain"
                )
            ]
            
            result = navigator.search_with_reasoning("test query")
            
            assert "query" in result
            assert "nodes_found" in result
            assert "nodes" in result
            assert "reasoning_stats" in result
            assert "reasoning_paths" in result
    
    def test_generate_path_explanation(self, graph_with_paths):
        navigator = GraphNavigator(graph_with_paths)
        
        node_path = ["n1", "n2", "n3"]
        edge_path = [
            ReasoningEdge("n1", "n2", EdgeType.CAUSE_EFFECT, 0.9),
            ReasoningEdge("n2", "n3", EdgeType.SUPPORT, 0.85)
        ]
        
        explanation = navigator._generate_path_explanation(node_path, edge_path)
        
        assert "Problem" in explanation
        assert "Cause" in explanation
        assert "Solution" in explanation
        assert "→" in explanation


class TestReasoningGraphExpansion:
    """Tests for path expansion in reasoning graph."""
    
    @pytest.fixture
    def chain_graph(self):
        graph = ReasoningGraph("chain_test")
        
        # Linear chain: A -> B -> C -> D
        for i, name in enumerate(["A", "B", "C", "D"]):
            graph.nodes[name] = GraphNode(name, f"Node {name}", f"Summary of {name}", depth=i)
        
        for src, tgt in [("A", "B"), ("B", "C"), ("C", "D")]:
            graph._add_edge(ReasoningEdge(src, tgt, EdgeType.TEMPORAL, 0.9))
        
        return graph
    
    def test_expand_from_node_finds_paths(self, chain_graph):
        paths = chain_graph._expand_from_node(
            start_id="A",
            initial_confidence=1.0,
            max_hops=4,
            min_confidence=0.5
        )
        
        # Should find paths: A->B, A->B->C, A->B->C->D
        assert len(paths) >= 3
        
        # Check path lengths
        path_lengths = sorted([len(p) for p in paths])
        assert 2 in path_lengths  # A->B
        assert 3 in path_lengths  # A->B->C
        assert 4 in path_lengths  # A->B->C->D
    
    def test_expand_respects_max_hops(self, chain_graph):
        paths = chain_graph._expand_from_node(
            start_id="A",
            initial_confidence=1.0,
            max_hops=2,
            min_confidence=0.5
        )
        
        # Max 2 nodes in path
        for path in paths:
            assert len(path) <= 2
    
    def test_expand_respects_min_confidence(self, chain_graph):
        # With high min_confidence, fewer paths should pass
        paths = chain_graph._expand_from_node(
            start_id="A",
            initial_confidence=1.0,
            max_hops=4,
            min_confidence=0.8
        )
        
        for path in paths:
            assert path.total_confidence >= 0.8
    
    def test_expand_no_cycles(self, chain_graph):
        # Add a back edge
        chain_graph._add_edge(ReasoningEdge("C", "A", EdgeType.REFERENCE, 0.5))
        
        paths = chain_graph._expand_from_node(
            start_id="A",
            initial_confidence=1.0,
            max_hops=5,
            min_confidence=0.3
        )
        
        # No path should have duplicate nodes
        for path in paths:
            node_ids = [n.id for n in path.nodes]
            assert len(node_ids) == len(set(node_ids))
