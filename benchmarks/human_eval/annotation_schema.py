"""
Human-evaluation annotation schema (PHASE 4-1 of the ACM upgrade plan).

Defines the dimensions annotators score, their scales and anchor descriptions,
and the inter-annotator agreement target used by the analysis scripts.
"""
from __future__ import annotations

ANNOTATION_DIMENSIONS = {
    "faithfulness": {
        "description": "Is every factual claim in the answer supported by the source document?",
        "scale": [1, 2, 3, 4, 5],
        "anchors": {
            1: "Answer contains fabricated information not in source",
            3: "Answer mostly accurate with minor unsupported claims",
            5: "Every claim is directly traceable to source document",
        },
    },
    "relevance": {
        "description": "Does the answer directly and completely address the question?",
        "scale": [1, 2, 3, 4, 5],
        "anchors": {
            1: "Answer is off-topic or misses the question entirely",
            3: "Answer partially addresses the question",
            5: "Answer directly and completely addresses the question",
        },
    },
    "citation_quality": {
        "description": "Are source citations specific, accurate, and useful?",
        "scale": [0, 1, 2],
        "anchors": {
            0: "No citations provided",
            1: "Citations present but vague (document-level only)",
            2: "Citations are page-specific and verifiable [Doc, p.X]",
        },
    },
}

INTER_ANNOTATOR_AGREEMENT_THRESHOLD = 0.6  # Krippendorff's alpha target

# Convenience: ordered list of dimension names and their score domains.
DIMENSION_NAMES = list(ANNOTATION_DIMENSIONS.keys())
SCORE_DOMAINS = {name: spec["scale"] for name, spec in ANNOTATION_DIMENSIONS.items()}


def is_valid_score(dimension: str, score) -> bool:
    """True if ``score`` is within the allowed scale for ``dimension``."""
    if dimension not in ANNOTATION_DIMENSIONS:
        return False
    try:
        return int(score) in ANNOTATION_DIMENSIONS[dimension]["scale"]
    except (TypeError, ValueError):
        return False
