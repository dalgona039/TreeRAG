"""
Learnable Scoring Function for TreeRAG.

Replaces fixed heuristic weights with learnable parameters:
- P(v|q) = σ(w₁·semantic + w₂·structural + w₃·contextual + b)

Features:
- Gradient-based weight optimization
- Pairwise ranking loss (BPR, Hinge)
- Cross-entropy loss for relevance prediction
- Support for different feature extractors
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum
from pathlib import Path


class LossType(str, Enum):
    """Types of loss functions."""
    CROSS_ENTROPY = "cross_entropy"
    BPR = "bpr"  # Bayesian Personalized Ranking
    HINGE = "hinge"
    MSE = "mse"
    LISTWISE = "listwise"


class FeatureType(str, Enum):
    """Types of features for scoring."""
    SEMANTIC = "semantic"  # Embedding similarity
    STRUCTURAL = "structural"  # Tree position
    CONTEXTUAL = "contextual"  # Query-context relevance
    LEXICAL = "lexical"  # BM25-like features
    POSITIONAL = "positional"  # Position in document


@dataclass
class ScoringFeatures:
    """Features extracted for a node."""
    node_id: str
    semantic_similarity: float  # Embedding cosine similarity
    structural_score: float  # Tree depth/position score
    contextual_overlap: float  # Query-context overlap
    lexical_score: float  # BM25/TF-IDF score
    positional_score: float  # Document position
    
    # Optional additional features
    parent_relevance: float = 0.0
    sibling_relevance: float = 0.0
    child_coverage: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector."""
        return [
            self.semantic_similarity,
            self.structural_score,
            self.contextual_overlap,
            self.lexical_score,
            self.positional_score,
            self.parent_relevance,
            self.sibling_relevance,
            self.child_coverage
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names."""
        return [
            "semantic_similarity",
            "structural_score",
            "contextual_overlap",
            "lexical_score",
            "positional_score",
            "parent_relevance",
            "sibling_relevance",
            "child_coverage"
        ]


@dataclass
class TrainingExample:
    """Single training example for relevance learning."""
    query_id: str
    query_text: str
    positive_node: ScoringFeatures  # Relevant node
    negative_nodes: List[ScoringFeatures]  # Non-relevant nodes
    relevance_scores: Optional[Dict[str, float]] = None  # Graded relevance


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    loss_type: LossType = LossType.BPR
    regularization: float = 0.01
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class EvaluationResult:
    """Result of evaluating the scoring function."""
    accuracy: float
    ndcg_at_5: float
    mrr: float
    loss: float
    weights: Dict[str, float]


class LearnableScoringFunction:
    """
    Learnable scoring function with gradient-based optimization.
    
    The scoring function computes:
    P(v|q) = σ(Σᵢ wᵢ·fᵢ(v,q) + b)
    
    Where:
    - wᵢ are learnable weights
    - fᵢ are feature functions
    - b is a bias term
    - σ is sigmoid activation
    """
    
    def __init__(
        self,
        n_features: int = 8,
        init_weights: Optional[List[float]] = None,
        random_seed: int = 42
    ):
        """
        Initialize learnable scoring function.
        
        Args:
            n_features: Number of input features
            init_weights: Initial weights (optional)
            random_seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Initialize weights
        if init_weights is not None:
            self.weights = list(init_weights)
        else:
            # Xavier initialization
            scale = math.sqrt(2.0 / n_features)
            self.weights = [random.gauss(0, scale) for _ in range(n_features)]
        
        # Bias term
        self.bias = 0.0
        
        # Gradient accumulators
        self._weight_grads = [0.0] * n_features
        self._bias_grad = 0.0
        
        # Training history
        self.training_history: List[Dict[str, float]] = []
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation with numerical stability."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)
    
    def score(self, features: ScoringFeatures) -> float:
        """
        Compute relevance score for a node.
        
        Args:
            features: Node features
            
        Returns:
            Probability of relevance (0 to 1)
        """
        feature_vector = features.to_vector()
        
        linear = sum(w * f for w, f in zip(self.weights, feature_vector))
        linear += self.bias
        
        return self.sigmoid(linear)
    
    def score_vector(self, feature_vector: List[float]) -> float:
        """Score from raw feature vector."""
        linear = sum(w * f for w, f in zip(self.weights, feature_vector))
        linear += self.bias
        return self.sigmoid(linear)
    
    def rank(self, candidates: List[ScoringFeatures]) -> List[Tuple[str, float]]:
        """
        Rank candidates by relevance score.
        
        Args:
            candidates: List of candidate features
            
        Returns:
            List of (node_id, score) sorted by score descending
        """
        scored = [(c.node_id, self.score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def compute_loss(
        self,
        example: TrainingExample,
        loss_type: LossType = LossType.BPR
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute loss and gradients for a training example.
        
        Args:
            example: Training example
            loss_type: Type of loss function
            
        Returns:
            Tuple of (loss value, gradients dict)
        """
        pos_features = example.positive_node.to_vector()
        pos_score = self.score_vector(pos_features)
        
        if loss_type == LossType.BPR:
            return self._bpr_loss(pos_features, pos_score, example)
        elif loss_type == LossType.HINGE:
            return self._hinge_loss(pos_features, pos_score, example)
        elif loss_type == LossType.CROSS_ENTROPY:
            return self._cross_entropy_loss(pos_features, pos_score, example)
        else:
            return self._mse_loss(pos_features, pos_score, example)
    
    def _bpr_loss(
        self,
        pos_features: List[float],
        pos_score: float,
        example: TrainingExample
    ) -> Tuple[float, Dict[str, float]]:
        """
        Bayesian Personalized Ranking loss.
        
        L = -log(σ(score_pos - score_neg))
        """
        total_loss = 0.0
        weight_grads = [0.0] * self.n_features
        bias_grad = 0.0
        
        for neg_node in example.negative_nodes:
            neg_features = neg_node.to_vector()
            neg_score = self.score_vector(neg_features)
            
            # BPR loss
            diff = pos_score - neg_score
            sig_diff = self.sigmoid(-diff)  # 1 - σ(diff) = σ(-diff)
            
            loss = -math.log(max(1 - sig_diff, 1e-10))
            total_loss += loss
            
            # Gradients
            # d_loss/d_w = -sig_diff * (pos_features - neg_features)
            for i in range(self.n_features):
                weight_grads[i] += sig_diff * (neg_features[i] - pos_features[i])
            
            bias_grad += sig_diff * (-1)
        
        n_neg = len(example.negative_nodes)
        if n_neg > 0:
            total_loss /= n_neg
            weight_grads = [g / n_neg for g in weight_grads]
            bias_grad /= n_neg
        
        return total_loss, {"weights": weight_grads, "bias": bias_grad}
    
    def _hinge_loss(
        self,
        pos_features: List[float],
        pos_score: float,
        example: TrainingExample
    ) -> Tuple[float, Dict[str, float]]:
        """
        Hinge loss with margin.
        
        L = max(0, margin - (score_pos - score_neg))
        """
        margin = 0.1
        total_loss = 0.0
        weight_grads = [0.0] * self.n_features
        bias_grad = 0.0
        
        for neg_node in example.negative_nodes:
            neg_features = neg_node.to_vector()
            neg_score = self.score_vector(neg_features)
            
            diff = pos_score - neg_score
            loss = max(0, margin - diff)
            total_loss += loss
            
            if loss > 0:
                # Gradients when margin is violated
                for i in range(self.n_features):
                    weight_grads[i] += (neg_features[i] - pos_features[i])
                bias_grad += -1
        
        n_neg = len(example.negative_nodes)
        if n_neg > 0:
            total_loss /= n_neg
            weight_grads = [g / n_neg for g in weight_grads]
            bias_grad /= n_neg
        
        return total_loss, {"weights": weight_grads, "bias": bias_grad}
    
    def _cross_entropy_loss(
        self,
        pos_features: List[float],
        pos_score: float,
        example: TrainingExample
    ) -> Tuple[float, Dict[str, float]]:
        """
        Binary cross-entropy loss.
        
        L = -[y·log(p) + (1-y)·log(1-p)]
        """
        # Positive example: y=1
        loss_pos = -math.log(max(pos_score, 1e-10))
        
        # Negative examples: y=0
        total_loss = loss_pos
        weight_grads = [0.0] * self.n_features
        bias_grad = 0.0
        
        # Gradient for positive
        grad_pos = pos_score - 1  # d_loss/d_score = (p - y)
        for i in range(self.n_features):
            weight_grads[i] += grad_pos * pos_features[i]
        bias_grad += grad_pos
        
        for neg_node in example.negative_nodes:
            neg_features = neg_node.to_vector()
            neg_score = self.score_vector(neg_features)
            
            loss_neg = -math.log(max(1 - neg_score, 1e-10))
            total_loss += loss_neg
            
            # Gradient for negative
            grad_neg = neg_score  # d_loss/d_score = p - 0 = p
            for i in range(self.n_features):
                weight_grads[i] += grad_neg * neg_features[i]
            bias_grad += grad_neg
        
        n_total = 1 + len(example.negative_nodes)
        total_loss /= n_total
        weight_grads = [g / n_total for g in weight_grads]
        bias_grad /= n_total
        
        return total_loss, {"weights": weight_grads, "bias": bias_grad}
    
    def _mse_loss(
        self,
        pos_features: List[float],
        pos_score: float,
        example: TrainingExample
    ) -> Tuple[float, Dict[str, float]]:
        """Mean squared error loss."""
        # Target: 1 for positive, 0 for negative
        loss_pos = (pos_score - 1) ** 2
        
        total_loss = loss_pos
        weight_grads = [0.0] * self.n_features
        bias_grad = 0.0
        
        # Gradient for positive
        grad_pos = 2 * (pos_score - 1) * pos_score * (1 - pos_score)
        for i in range(self.n_features):
            weight_grads[i] += grad_pos * pos_features[i]
        bias_grad += grad_pos
        
        for neg_node in example.negative_nodes:
            neg_features = neg_node.to_vector()
            neg_score = self.score_vector(neg_features)
            
            loss_neg = neg_score ** 2
            total_loss += loss_neg
            
            grad_neg = 2 * neg_score * neg_score * (1 - neg_score)
            for i in range(self.n_features):
                weight_grads[i] += grad_neg * neg_features[i]
            bias_grad += grad_neg
        
        n_total = 1 + len(example.negative_nodes)
        total_loss /= n_total
        weight_grads = [g / n_total for g in weight_grads]
        bias_grad /= n_total
        
        return total_loss, {"weights": weight_grads, "bias": bias_grad}
    
    def update_weights(
        self,
        gradients: Dict[str, Any],
        learning_rate: float,
        regularization: float = 0.0
    ) -> None:
        """
        Update weights using gradients.
        
        Args:
            gradients: Weight and bias gradients
            learning_rate: Learning rate
            regularization: L2 regularization strength
        """
        weight_grads = gradients["weights"]
        bias_grad = gradients["bias"]
        
        # Update weights with regularization
        for i in range(self.n_features):
            reg_term = regularization * self.weights[i]
            self.weights[i] -= learning_rate * (weight_grads[i] + reg_term)
        
        # Update bias
        self.bias -= learning_rate * bias_grad
    
    def train(
        self,
        examples: List[TrainingExample],
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Train the scoring function.
        
        Args:
            examples: Training examples
            config: Training configuration
            
        Returns:
            Training results
        """
        random.seed(config.random_seed)
        
        # Split data
        n_val = int(len(examples) * config.validation_split)
        random.shuffle(examples)
        val_examples = examples[:n_val]
        train_examples = examples[n_val:]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Shuffle training data
            random.shuffle(train_examples)
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(train_examples), config.batch_size):
                batch = train_examples[i:i + config.batch_size]
                
                batch_grads = {"weights": [0.0] * self.n_features, "bias": 0.0}
                batch_loss = 0.0
                
                for example in batch:
                    loss, grads = self.compute_loss(example, config.loss_type)
                    batch_loss += loss
                    
                    for j in range(self.n_features):
                        batch_grads["weights"][j] += grads["weights"][j]
                    batch_grads["bias"] += grads["bias"]
                
                # Average gradients
                batch_size = len(batch)
                batch_grads["weights"] = [g / batch_size for g in batch_grads["weights"]]
                batch_grads["bias"] /= batch_size
                
                # Update weights
                self.update_weights(batch_grads, config.learning_rate, config.regularization)
                
                epoch_loss += batch_loss / batch_size
                n_batches += 1
            
            epoch_loss /= n_batches
            
            # Validation
            val_loss = self._evaluate_loss(val_examples, config.loss_type)
            
            # Record history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "weights": list(self.weights),
                "bias": self.bias
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    break
        
        return {
            "final_weights": dict(zip(ScoringFeatures.feature_names(), self.weights)),
            "final_bias": self.bias,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.training_history),
            "history": self.training_history
        }
    
    def _evaluate_loss(
        self,
        examples: List[TrainingExample],
        loss_type: LossType
    ) -> float:
        """Evaluate total loss on examples."""
        if not examples:
            return 0.0
        
        total_loss = 0.0
        for example in examples:
            loss, _ = self.compute_loss(example, loss_type)
            total_loss += loss
        
        return total_loss / len(examples)
    
    def evaluate(
        self,
        test_examples: List[TrainingExample]
    ) -> EvaluationResult:
        """
        Evaluate scoring function on test data.
        
        Args:
            test_examples: Test examples
            
        Returns:
            Evaluation metrics
        """
        if not test_examples:
            return EvaluationResult(0, 0, 0, 0, {})
        
        correct = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        
        for example in test_examples:
            pos_score = self.score(example.positive_node)
            
            # Check if positive ranks first
            all_nodes = [example.positive_node] + example.negative_nodes
            ranked = self.rank(all_nodes)
            
            # Accuracy: positive node ranked first
            if ranked[0][0] == example.positive_node.node_id:
                correct += 1
            
            # MRR
            for i, (node_id, _) in enumerate(ranked):
                if node_id == example.positive_node.node_id:
                    mrr_sum += 1.0 / (i + 1)
                    break
            
            # NDCG@5
            ndcg = self._compute_ndcg(ranked[:5], example.positive_node.node_id)
            ndcg_sum += ndcg
        
        n = len(test_examples)
        loss = self._evaluate_loss(test_examples, LossType.BPR)
        
        return EvaluationResult(
            accuracy=correct / n,
            ndcg_at_5=ndcg_sum / n,
            mrr=mrr_sum / n,
            loss=loss,
            weights=dict(zip(ScoringFeatures.feature_names(), self.weights))
        )
    
    def _compute_ndcg(
        self,
        ranked: List[Tuple[str, float]],
        positive_id: str
    ) -> float:
        """Compute NDCG for ranked results."""
        dcg = 0.0
        for i, (node_id, _) in enumerate(ranked):
            if node_id == positive_id:
                dcg = 1.0 / math.log2(i + 2)
                break
        
        # Ideal DCG: positive at rank 1
        idcg = 1.0 / math.log2(2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "n_features": self.n_features,
            "weights": self.weights,
            "bias": self.bias,
            "feature_names": ScoringFeatures.feature_names(),
            "training_history": self.training_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "LearnableScoringFunction":
        """Load model from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model = cls(
            n_features=data["n_features"],
            init_weights=data["weights"]
        )
        model.bias = data["bias"]
        model.training_history = data.get("training_history", [])
        
        return model
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on absolute weights."""
        names = ScoringFeatures.feature_names()
        abs_weights = [abs(w) for w in self.weights]
        total = sum(abs_weights)
        
        if total == 0:
            return {name: 0.0 for name in names}
        
        return {
            name: abs_w / total
            for name, abs_w in zip(names, abs_weights)
        }


class FeatureExtractor:
    """
    Extract features for scoring from nodes and queries.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """Initialize feature extractor."""
        self.embedding_model = embedding_model
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def extract(
        self,
        node_id: str,
        node_text: str,
        query_text: str,
        tree_depth: int = 0,
        tree_max_depth: int = 5,
        position_in_doc: float = 0.0,
        parent_node: Optional[Any] = None,
        siblings: Optional[List[Any]] = None
    ) -> ScoringFeatures:
        """
        Extract features for a node given a query.
        
        Args:
            node_id: Node identifier
            node_text: Node content text
            query_text: Query text
            tree_depth: Depth of node in tree
            tree_max_depth: Maximum tree depth
            position_in_doc: Position in document (0-1)
            parent_node: Parent node for context
            siblings: Sibling nodes for context
            
        Returns:
            Extracted features
        """
        # Semantic similarity (placeholder - would use embeddings)
        semantic = self._compute_semantic_similarity(node_text, query_text)
        
        # Structural score based on depth (middle levels often best)
        structural = self._compute_structural_score(tree_depth, tree_max_depth)
        
        # Contextual overlap
        contextual = self._compute_contextual_overlap(node_text, query_text)
        
        # Lexical score (BM25-like)
        lexical = self._compute_lexical_score(node_text, query_text)
        
        # Positional score
        positional = 1.0 - position_in_doc  # Earlier is often better
        
        # Parent relevance
        parent_rel = 0.0
        if parent_node:
            parent_rel = self._compute_semantic_similarity(
                str(parent_node), query_text
            )
        
        # Sibling relevance (average)
        sibling_rel = 0.0
        if siblings:
            sibling_scores = [
                self._compute_semantic_similarity(str(s), query_text)
                for s in siblings
            ]
            sibling_rel = sum(sibling_scores) / len(sibling_scores)
        
        return ScoringFeatures(
            node_id=node_id,
            semantic_similarity=semantic,
            structural_score=structural,
            contextual_overlap=contextual,
            lexical_score=lexical,
            positional_score=positional,
            parent_relevance=parent_rel,
            sibling_relevance=sibling_rel,
            child_coverage=0.0  # Would need children info
        )
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity (placeholder)."""
        # In practice, use embeddings
        # Here we use word overlap as proxy
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_structural_score(self, depth: int, max_depth: int) -> float:
        """Compute structural score based on tree position."""
        if max_depth == 0:
            return 0.5
        
        # Penalize very shallow (too general) and very deep (too specific)
        optimal_depth = max_depth * 0.4  # Sweet spot at 40% depth
        distance = abs(depth - optimal_depth)
        
        return max(0, 1 - distance / max_depth)
    
    def _compute_contextual_overlap(self, node_text: str, query_text: str) -> float:
        """Compute contextual overlap."""
        query_words = set(query_text.lower().split())
        node_words = set(node_text.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & node_words)
        return overlap / len(query_words)
    
    def _compute_lexical_score(self, node_text: str, query_text: str) -> float:
        """Compute BM25-like lexical score."""
        # Simplified BM25
        k1, b = 1.2, 0.75
        avg_len = 100  # Assumed average document length
        
        query_terms = query_text.lower().split()
        node_terms = node_text.lower().split()
        doc_len = len(node_terms)
        
        if not query_terms or not node_terms:
            return 0.0
        
        score = 0.0
        node_term_set = set(node_terms)
        
        for term in query_terms:
            if term in node_term_set:
                tf = node_terms.count(term)
                norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
                score += norm_tf
        
        # Normalize
        max_possible = len(query_terms) * (k1 + 1)
        return score / max_possible if max_possible > 0 else 0.0


def create_training_data_from_labeled(
    queries: List[Dict[str, Any]],
    nodes: Dict[str, Dict[str, Any]],
    extractor: FeatureExtractor
) -> List[TrainingExample]:
    """
    Create training data from labeled query-node pairs.
    
    Args:
        queries: List of queries with relevant_nodes field
        nodes: Dict of node_id -> node data
        extractor: Feature extractor
        
    Returns:
        List of training examples
    """
    examples = []
    
    for query in queries:
        query_id = query["id"]
        query_text = query["text"]
        relevant_ids = set(query.get("relevant_nodes", []))
        
        if not relevant_ids:
            continue
        
        # Extract features for all nodes
        node_features = {}
        for node_id, node_data in nodes.items():
            features = extractor.extract(
                node_id=node_id,
                node_text=node_data.get("text", ""),
                query_text=query_text,
                tree_depth=node_data.get("depth", 0),
                tree_max_depth=node_data.get("max_depth", 5),
                position_in_doc=node_data.get("position", 0.0)
            )
            node_features[node_id] = features
        
        # Create examples: each relevant node paired with negatives
        for pos_id in relevant_ids:
            if pos_id not in node_features:
                continue
            
            pos_features = node_features[pos_id]
            neg_features = [
                f for nid, f in node_features.items()
                if nid not in relevant_ids
            ]
            
            if neg_features:
                examples.append(TrainingExample(
                    query_id=query_id,
                    query_text=query_text,
                    positive_node=pos_features,
                    negative_nodes=neg_features[:10]  # Limit negatives
                ))
    
    return examples
