"""
Beam Search Algorithm Tests

Beam Search 구현을 검증하는 테스트 스위트
"""

import pytest
from typing import Dict, Any


# ===== Test Fixtures =====

@pytest.fixture
def sample_tree() -> Dict[str, Any]:
    """테스트용 샘플 트리 구조"""
    return {
        "id": "root",
        "title": "졸업요건 안내서",
        "summary": "학과별 졸업 요건과 학점 구조를 설명하는 문서",
        "page_ref": "p.1",
        "children": [
            {
                "id": "ch1",
                "title": "전공 필수 과목",
                "summary": "전공 필수로 이수해야 하는 과목 목록",
                "page_ref": "p.5-10",
                "children": [
                    {
                        "id": "ch1_1",
                        "title": "프로그래밍 기초",
                        "summary": "Python 및 Java 프로그래밍 기초 과목",
                        "page_ref": "p.5"
                    },
                    {
                        "id": "ch1_2",
                        "title": "자료구조",
                        "summary": "자료구조와 알고리즘의 기초",
                        "page_ref": "p.6"
                    },
                    {
                        "id": "ch1_3",
                        "title": "운영체제",
                        "summary": "운영체제 원리와 시스템 프로그래밍",
                        "page_ref": "p.7"
                    }
                ]
            },
            {
                "id": "ch2",
                "title": "전공 선택 과목",
                "summary": "전공 선택으로 이수할 수 있는 과목 목록",
                "page_ref": "p.11-20",
                "children": [
                    {
                        "id": "ch2_1",
                        "title": "인공지능",
                        "summary": "머신러닝과 딥러닝 기초",
                        "page_ref": "p.11"
                    },
                    {
                        "id": "ch2_2",
                        "title": "데이터베이스",
                        "summary": "관계형 데이터베이스와 SQL",
                        "page_ref": "p.12"
                    }
                ]
            },
            {
                "id": "ch3",
                "title": "교양 과목",
                "summary": "필수 교양 및 선택 교양 과목",
                "page_ref": "p.21-30",
                "children": [
                    {
                        "id": "ch3_1",
                        "title": "영어",
                        "summary": "영어 회화 및 작문",
                        "page_ref": "p.21"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def deep_tree() -> Dict[str, Any]:
    """깊은 트리 구조 (depth=4)"""
    return {
        "id": "root",
        "title": "Level 0",
        "summary": "Root node",
        "children": [
            {
                "id": f"L1_{i}",
                "title": f"Level 1 - Item {i}",
                "summary": f"First level item {i}",
                "children": [
                    {
                        "id": f"L2_{i}_{j}",
                        "title": f"Level 2 - Item {i}.{j}",
                        "summary": f"Second level item {i}.{j}" + (" 인공지능" if i == 0 and j == 1 else ""),
                        "children": [
                            {
                                "id": f"L3_{i}_{j}_{k}",
                                "title": f"Level 3 - Item {i}.{j}.{k}",
                                "summary": f"Third level item {i}.{j}.{k}" + (" 딥러닝 관련" if i == 0 and j == 1 else "")
                            }
                            for k in range(2)
                        ]
                    }
                    for j in range(3)
                ]
            }
            for i in range(4)
        ]
    }


# ===== BeamNode Tests =====

class TestBeamNode:
    """BeamNode 데이터 클래스 테스트"""
    
    def test_create_beam_node(self):
        """BeamNode 생성 테스트"""
        from src.core.beam_search import BeamNode
        
        node = {"id": "test", "title": "Test Node"}
        beam_node = BeamNode.create(
            node=node,
            path="Root > Test Node",
            depth=1,
            score=0.8,
            parent_context="Root"
        )
        
        assert beam_node.node == node
        assert beam_node.path == "Root > Test Node"
        assert beam_node.depth == 1
        assert beam_node.cumulative_score == 0.8
        assert beam_node.priority == -0.8  # Negative for max-heap
    
    def test_beam_node_ordering(self):
        """BeamNode 정렬 테스트 (높은 점수가 우선)"""
        from src.core.beam_search import BeamNode
        
        nodes = [
            BeamNode.create({"id": "1"}, "p1", 0, 0.5),
            BeamNode.create({"id": "2"}, "p2", 0, 0.9),
            BeamNode.create({"id": "3"}, "p3", 0, 0.3),
        ]
        
        sorted_nodes = sorted(nodes)
        
        # priority가 음수이므로, 낮은 priority(높은 점수)가 먼저
        assert sorted_nodes[0].cumulative_score == 0.9
        assert sorted_nodes[1].cumulative_score == 0.5
        assert sorted_nodes[2].cumulative_score == 0.3


# ===== BeamSearchNavigator Tests =====

class TestBeamSearchNavigator:
    """BeamSearchNavigator 테스트"""
    
    def test_initialization(self, sample_tree):
        """Navigator 초기화 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        navigator = BeamSearchNavigator(
            tree=sample_tree,
            document_name="test_doc",
            beam_width=3
        )
        
        assert navigator.document_name == "test_doc"
        assert navigator.beam_width == 3
        assert navigator.tree == sample_tree
    
    def test_keyword_score(self, sample_tree):
        """키워드 점수 계산 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        navigator = BeamSearchNavigator(sample_tree, "test_doc")
        
        # 관련 키워드가 있는 노드
        node = {"title": "인공지능", "summary": "머신러닝과 딥러닝"}
        score = navigator._keyword_score(node, "인공지능 딥러닝")
        assert score > 0.5
        
        # 관련 없는 노드
        node_irrelevant = {"title": "영어", "summary": "영어 회화"}
        score_irrelevant = navigator._keyword_score(node_irrelevant, "인공지능 딥러닝")
        assert score_irrelevant < score
    
    def test_search_returns_results(self, sample_tree):
        """검색이 결과를 반환하는지 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        navigator = BeamSearchNavigator(sample_tree, "test_doc", beam_width=3)
        
        results, stats = navigator.search(
            query="인공지능 과목",
            max_depth=3,
            min_score_threshold=0.1  # 낮은 임계값으로 결과 보장
        )
        
        assert isinstance(results, list)
        assert isinstance(stats, dict)
        assert "algorithm" in stats
        assert stats["algorithm"] == "beam_search"
        assert "nodes_selected" in stats
    
    def test_search_stats_format(self, sample_tree):
        """검색 통계 형식 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        navigator = BeamSearchNavigator(sample_tree, "test_doc", beam_width=5)
        
        _, stats = navigator.search("프로그래밍", max_depth=2)
        
        required_keys = [
            "algorithm", "beam_width", "nodes_visited", 
            "nodes_evaluated", "nodes_expanded", "nodes_selected", "max_depth_used"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        assert stats["beam_width"] == 5
        assert stats["max_depth_used"] == 2
    
    def test_beam_width_limits_nodes(self, deep_tree):
        """Beam width가 노드 수를 제한하는지 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        # 좁은 beam
        nav_narrow = BeamSearchNavigator(deep_tree, "doc", beam_width=2)
        _, stats_narrow = nav_narrow.search("테스트", max_depth=3)
        
        # 넓은 beam
        nav_wide = BeamSearchNavigator(deep_tree, "doc", beam_width=10)
        _, stats_wide = nav_wide.search("테스트", max_depth=3)
        
        # 좁은 beam이 더 적은 노드를 확장해야 함
        assert stats_narrow["nodes_expanded"] <= stats_wide["nodes_expanded"]
    
    def test_score_threshold_filters_results(self, sample_tree):
        """점수 임계값이 결과를 필터링하는지 테스트"""
        from src.core.beam_search import BeamSearchNavigator
        
        navigator = BeamSearchNavigator(sample_tree, "doc", beam_width=5)
        
        # 낮은 임계값
        results_low, _ = navigator.search("테스트", min_score_threshold=0.1)
        
        # 높은 임계값
        navigator2 = BeamSearchNavigator(sample_tree, "doc", beam_width=5)
        results_high, _ = navigator2.search("테스트", min_score_threshold=0.8)
        
        # 높은 임계값이 더 적은 결과를 반환해야 함
        assert len(results_high) <= len(results_low)


# ===== Integration with Reasoner Tests =====

class TestReasonerIntegration:
    """Reasoner와의 통합 테스트"""
    
    def test_reasoner_accepts_algorithm_param(self):
        """Reasoner가 알고리즘 파라미터를 받는지 테스트"""
        from src.core.reasoner import TreeRAGReasoner
        
        # 파일이 없으면 스킵
        import os
        from src.config import Config
        
        indices = [f for f in os.listdir(Config.INDEX_DIR) if f.endswith("_index.json")]
        if not indices:
            pytest.skip("No index files available")
        
        # Beam Search 모드로 초기화
        reasoner = TreeRAGReasoner(
            index_filenames=[indices[0]],
            use_deep_traversal=True,
            traversal_algorithm="beam_search",
            beam_width=5
        )
        
        assert reasoner.traversal_algorithm == "beam_search"
        assert reasoner.beam_width == 5
    
    def test_reasoner_default_is_beam_search(self):
        """기본 알고리즘이 Beam Search인지 테스트"""
        from src.core.reasoner import TreeRAGReasoner
        
        import os
        from src.config import Config
        
        indices = [f for f in os.listdir(Config.INDEX_DIR) if f.endswith("_index.json")]
        if not indices:
            pytest.skip("No index files available")
        
        reasoner = TreeRAGReasoner(
            index_filenames=[indices[0]],
            use_deep_traversal=True
        )
        
        assert reasoner.traversal_algorithm == "beam_search"


# ===== Formatting Tests =====

class TestFormatting:
    """결과 포맷팅 테스트"""
    
    def test_format_beam_results_empty(self):
        """빈 결과 포맷팅"""
        from src.core.beam_search import format_beam_results
        
        result = format_beam_results([], "test_doc")
        assert "No relevant sections found" in result
    
    def test_format_beam_results_with_data(self):
        """데이터가 있는 결과 포맷팅"""
        from src.core.beam_search import format_beam_results
        
        results = [
            {
                "node": {"title": "Test", "page_ref": "p.5", "summary": "Summary"},
                "path": "Root > Test",
                "depth": 1,
                "score": 0.85
            }
        ]
        
        formatted = format_beam_results(results, "test_doc")
        
        assert "test_doc" in formatted
        assert "Test" in formatted
        assert "0.85" in formatted
        assert "Beam Search" in formatted
