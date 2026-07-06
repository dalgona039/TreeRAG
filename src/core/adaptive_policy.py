"""Query-adaptive traversal-selection criterion.

Decides DFS vs. Beam Search from the score spread across a query's depth-1
candidate subtrees: a dominant top-1 candidate favors exhaustive DFS down a
narrow path, while spread-out scores favor Beam Search's wider coverage.
"""
import re
from typing import Any, Dict, List


def _keyword_overlap_score(text: str, query: str) -> float:
    query_keywords = {kw for kw in re.findall(r"\w+", query.lower()) if len(kw) > 2}
    if not query_keywords:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for kw in query_keywords if kw in text_lower)
    return matched / len(query_keywords)


def score_root_children(tree: Dict[str, Any], query: str) -> List[float]:
    """Cheap keyword-overlap score for each of the tree's depth-1 children.

    Used only to decide DFS vs. Beam before the real (possibly LLM-backed)
    traversal begins, so it must not itself call an LLM.
    """
    children = tree.get("children", []) or []
    scores = []
    for child in children:
        text = f"{child.get('title', '')} {child.get('summary', '')}"
        scores.append(_keyword_overlap_score(text, query))
    return scores


def choose_traversal_algorithm(
    root_children_scores: List[float], margin_cutoff: float = 0.15,
) -> str:
    """Return 'dfs' when the top candidate dominates the runner-up by
    >= margin_cutoff (low ambiguity -> precision-favoring exhaustive
    search); otherwise 'beam_search' (score mass spread across several
    subtrees -> coverage-favoring pruned search)."""
    if len(root_children_scores) < 2:
        return "dfs"
    ranked = sorted(root_children_scores, reverse=True)
    margin = ranked[0] - ranked[1]
    return "dfs" if margin >= margin_cutoff else "beam_search"
