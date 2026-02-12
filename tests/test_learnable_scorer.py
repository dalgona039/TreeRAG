"""
Tests for Learnable Scoring Function.

Tests:
- Feature extraction
- Scoring computation
- Loss functions (BPR, Hinge, CrossEntropy, MSE)
- Weight updates and training
- Model save/load
- Feature importance
"""

import pytest
import math
from typing import List
from pathlib import Path
import tempfile
import json

from src.core.learnable_scorer import (
    LearnableScoringFunction,
    ScoringFeatures,
    TrainingExample,
    TrainingConfig,
    LossType,
    FeatureExtractor,
    EvaluationResult,
    create_training_data_from_labeled
)


class TestScoringFeatures:
    """Tests for ScoringFeatures dataclass."""
    
    def test_to_vector(self):
        """Test feature vector conversion."""
        features = ScoringFeatures(
            node_id="n1",
            semantic_similarity=0.8,
            structural_score=0.6,
            contextual_overlap=0.7,
            lexical_score=0.5,
            positional_score=0.9,
            parent_relevance=0.3,
            sibling_relevance=0.4,
            child_coverage=0.2
        )
        
        vector = features.to_vector()
        
        assert len(vector) == 8
        assert vector[0] == 0.8  # semantic
        assert vector[1] == 0.6  # structural
        assert vector[4] == 0.9  # positional
    
    def test_feature_names(self):
        """Test feature names."""
        names = ScoringFeatures.feature_names()
        
        assert len(names) == 8
        assert "semantic_similarity" in names
        assert "structural_score" in names


class TestLearnableScoringFunction:
    """Tests for LearnableScoringFunction."""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer with default initialization."""
        return LearnableScoringFunction(n_features=8, random_seed=42)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample scoring features."""
        return ScoringFeatures(
            node_id="n1",
            semantic_similarity=0.8,
            structural_score=0.6,
            contextual_overlap=0.7,
            lexical_score=0.5,
            positional_score=0.9,
            parent_relevance=0.3,
            sibling_relevance=0.4,
            child_coverage=0.2
        )
    
    @pytest.fixture
    def training_example(self, sample_features):
        """Create sample training example."""
        neg_features = [
            ScoringFeatures(
                node_id="n2",
                semantic_similarity=0.2,
                structural_score=0.3,
                contextual_overlap=0.1,
                lexical_score=0.2,
                positional_score=0.5,
                parent_relevance=0.1,
                sibling_relevance=0.1,
                child_coverage=0.1
            ),
            ScoringFeatures(
                node_id="n3",
                semantic_similarity=0.3,
                structural_score=0.4,
                contextual_overlap=0.2,
                lexical_score=0.3,
                positional_score=0.4,
                parent_relevance=0.2,
                sibling_relevance=0.2,
                child_coverage=0.1
            )
        ]
        
        return TrainingExample(
            query_id="q1",
            query_text="test query",
            positive_node=sample_features,
            negative_nodes=neg_features
        )
    
    def test_sigmoid(self, scorer):
        """Test sigmoid activation."""
        assert abs(scorer.sigmoid(0) - 0.5) < 0.001
        assert scorer.sigmoid(10) > 0.99
        assert scorer.sigmoid(-10) < 0.01
    
    def test_score(self, scorer, sample_features):
        """Test scoring computation."""
        score = scorer.score(sample_features)
        
        # Score should be between 0 and 1 (sigmoid output)
        assert 0 <= score <= 1
    
    def test_rank(self, scorer):
        """Test candidate ranking."""
        candidates = [
            ScoringFeatures(
                node_id="n1", semantic_similarity=0.9,
                structural_score=0.8, contextual_overlap=0.7,
                lexical_score=0.6, positional_score=0.5,
                parent_relevance=0.4, sibling_relevance=0.3, child_coverage=0.2
            ),
            ScoringFeatures(
                node_id="n2", semantic_similarity=0.3,
                structural_score=0.2, contextual_overlap=0.1,
                lexical_score=0.1, positional_score=0.1,
                parent_relevance=0.1, sibling_relevance=0.1, child_coverage=0.1
            )
        ]
        
        ranked = scorer.rank(candidates)
        
        assert len(ranked) == 2
        # Higher scoring node should rank first
        assert ranked[0][1] >= ranked[1][1]
    
    def test_bpr_loss(self, scorer, training_example):
        """Test BPR loss computation."""
        loss, grads = scorer.compute_loss(training_example, LossType.BPR)
        
        assert loss >= 0
        assert "weights" in grads
        assert "bias" in grads
        assert len(grads["weights"]) == 8
    
    def test_hinge_loss(self, scorer, training_example):
        """Test hinge loss computation."""
        loss, grads = scorer.compute_loss(training_example, LossType.HINGE)
        
        assert loss >= 0
        assert "weights" in grads
    
    def test_cross_entropy_loss(self, scorer, training_example):
        """Test cross-entropy loss computation."""
        loss, grads = scorer.compute_loss(training_example, LossType.CROSS_ENTROPY)
        
        assert loss >= 0
        assert "weights" in grads
    
    def test_mse_loss(self, scorer, training_example):
        """Test MSE loss computation."""
        loss, grads = scorer.compute_loss(training_example, LossType.MSE)
        
        assert loss >= 0
        assert "weights" in grads
    
    def test_weight_update(self, scorer, training_example):
        """Test weight updates."""
        initial_weights = list(scorer.weights)
        
        loss, grads = scorer.compute_loss(training_example, LossType.BPR)
        scorer.update_weights(grads, learning_rate=0.1)
        
        # Weights should have changed
        weights_changed = any(
            abs(w1 - w2) > 0.0001
            for w1, w2 in zip(initial_weights, scorer.weights)
        )
        assert weights_changed
    
    def test_training(self, scorer, training_example):
        """Test full training loop."""
        examples = [training_example] * 10  # Replicate for batch
        
        config = TrainingConfig(
            learning_rate=0.1,
            batch_size=5,
            epochs=10,
            loss_type=LossType.BPR,
            regularization=0.001,
            early_stopping_patience=5,
            validation_split=0.2
        )
        
        result = scorer.train(examples, config)
        
        assert "final_weights" in result
        assert "final_bias" in result
        assert "best_val_loss" in result
        assert len(scorer.training_history) > 0
    
    def test_evaluation(self, scorer, training_example):
        """Test model evaluation."""
        examples = [training_example]
        
        result = scorer.evaluate(examples)
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.ndcg_at_5 <= 1
        assert 0 <= result.mrr <= 1
    
    def test_save_load(self, scorer, sample_features):
        """Test model save and load."""
        # Score before save
        score_before = scorer.score(sample_features)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            scorer.save(temp_path)
            
            # Load model
            loaded = LearnableScoringFunction.load(temp_path)
            
            # Score should be the same
            score_after = loaded.score(sample_features)
            
            assert abs(score_before - score_after) < 0.0001
            assert loaded.weights == scorer.weights
            assert loaded.bias == scorer.bias
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_feature_importance(self, scorer):
        """Test feature importance calculation."""
        importance = scorer.get_feature_importance()
        
        assert len(importance) == 8
        assert all(0 <= v <= 1 for v in importance.values())
        
        # Should sum to approximately 1
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.001 or total == 0
    
    def test_custom_initialization(self):
        """Test custom weight initialization."""
        init_weights = [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        scorer = LearnableScoringFunction(
            n_features=8,
            init_weights=init_weights
        )
        
        assert scorer.weights == init_weights


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor."""
        return FeatureExtractor()
    
    def test_extract_features(self, extractor):
        """Test feature extraction."""
        features = extractor.extract(
            node_id="n1",
            node_text="This is a test document about machine learning.",
            query_text="What is machine learning?",
            tree_depth=2,
            tree_max_depth=5,
            position_in_doc=0.3
        )
        
        assert features.node_id == "n1"
        assert 0 <= features.semantic_similarity <= 1
        assert 0 <= features.structural_score <= 1
        assert 0 <= features.contextual_overlap <= 1
        assert 0 <= features.lexical_score <= 1
        assert 0 <= features.positional_score <= 1
    
    def test_semantic_similarity(self, extractor):
        """Test semantic similarity computation."""
        sim_high = extractor._compute_semantic_similarity(
            "machine learning deep neural networks",
            "deep learning neural networks"
        )
        
        sim_low = extractor._compute_semantic_similarity(
            "cooking recipes food ingredients",
            "machine learning algorithms"
        )
        
        assert sim_high > sim_low
    
    def test_structural_score(self, extractor):
        """Test structural score computation."""
        # Middle depth should score highest
        score_shallow = extractor._compute_structural_score(0, 10)
        score_middle = extractor._compute_structural_score(4, 10)
        score_deep = extractor._compute_structural_score(10, 10)
        
        assert score_middle >= score_shallow
        assert score_middle >= score_deep
    
    def test_lexical_score(self, extractor):
        """Test lexical score computation."""
        score_match = extractor._compute_lexical_score(
            "machine learning algorithms are powerful",
            "machine learning"
        )
        
        score_no_match = extractor._compute_lexical_score(
            "cooking is fun and easy",
            "machine learning"
        )
        
        assert score_match > score_no_match
    
    def test_with_parent_context(self, extractor):
        """Test feature extraction with parent context."""
        features = extractor.extract(
            node_id="n1",
            node_text="Neural networks are used.",
            query_text="What are neural networks?",
            tree_depth=3,
            tree_max_depth=5,
            parent_node="Machine learning includes neural networks."
        )
        
        assert features.parent_relevance > 0


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.loss_type == LossType.BPR
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=64,
            epochs=50,
            loss_type=LossType.HINGE
        )
        
        assert config.learning_rate == 0.001
        assert config.loss_type == LossType.HINGE


class TestCreateTrainingData:
    """Tests for training data creation."""
    
    def test_create_from_labeled(self):
        """Test creating training data from labeled queries."""
        queries = [
            {
                "id": "q1",
                "text": "What is machine learning?",
                "relevant_nodes": ["n1", "n3"]
            }
        ]
        
        nodes = {
            "n1": {"text": "Machine learning overview.", "depth": 1, "max_depth": 5},
            "n2": {"text": "Cooking recipes.", "depth": 2, "max_depth": 5},
            "n3": {"text": "ML algorithms explained.", "depth": 2, "max_depth": 5}
        }
        
        extractor = FeatureExtractor()
        examples = create_training_data_from_labeled(queries, nodes, extractor)
        
        assert len(examples) == 2  # Two relevant nodes = two examples
        
        # Each example should have positive and negative nodes
        for example in examples:
            assert example.positive_node is not None
            assert len(example.negative_nodes) > 0
            assert example.positive_node.node_id in ["n1", "n3"]


class TestIntegration:
    """Integration tests for full scoring pipeline."""
    
    def test_full_pipeline(self):
        """Test complete training and evaluation pipeline."""
        # Create sample data
        extractor = FeatureExtractor()
        
        # Generate examples
        examples = []
        for i in range(20):
            pos_features = ScoringFeatures(
                node_id=f"pos_{i}",
                semantic_similarity=0.7 + 0.3 * (i % 2),
                structural_score=0.6,
                contextual_overlap=0.8,
                lexical_score=0.5,
                positional_score=0.7,
                parent_relevance=0.4,
                sibling_relevance=0.3,
                child_coverage=0.2
            )
            
            neg_features = [
                ScoringFeatures(
                    node_id=f"neg_{i}_{j}",
                    semantic_similarity=0.2,
                    structural_score=0.3,
                    contextual_overlap=0.1,
                    lexical_score=0.2,
                    positional_score=0.3,
                    parent_relevance=0.1,
                    sibling_relevance=0.1,
                    child_coverage=0.1
                )
                for j in range(3)
            ]
            
            examples.append(TrainingExample(
                query_id=f"q_{i}",
                query_text=f"test query {i}",
                positive_node=pos_features,
                negative_nodes=neg_features
            ))
        
        # Train model
        scorer = LearnableScoringFunction(n_features=8, random_seed=42)
        config = TrainingConfig(
            learning_rate=0.1,
            batch_size=10,
            epochs=20,
            loss_type=LossType.BPR,
            early_stopping_patience=5
        )
        
        result = scorer.train(examples[:15], config)
        
        # Evaluate
        eval_result = scorer.evaluate(examples[15:])
        
        # Model should perform reasonably well on this easy data
        assert eval_result.accuracy >= 0.5
        assert eval_result.mrr >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
