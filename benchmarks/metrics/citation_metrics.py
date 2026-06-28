"""
Citation traceability and accuracy metrics.

citation_availability  – fraction of questions where at least one retrieved node
                         carries a non-empty page_ref.
section_citation_f1    – F1 between retrieved section IDs and expected_sections
                         using ancestor/descendant matching.  Returns None when
                         the question has no expected_sections (shouldn't happen
                         in full_benchmark, but is graceful on other datasets).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
def _node_id(n: Dict[str, Any]) -> str:
    """Extract node ID from a possibly-nested node dict."""
    if "id" in n and n["id"]:
        return str(n["id"])
    sub = n.get("node") or {}
    return str(sub.get("id", "")) if isinstance(sub, dict) else ""


def _page_ref(n: Dict[str, Any]) -> str:
    """Extract page_ref from a possibly-nested node dict."""
    pr = n.get("page_ref", "")
    if pr:
        return str(pr)
    sub = n.get("node") or {}
    if isinstance(sub, dict):
        return str(sub.get("page_ref", ""))
    return ""


def citation_availability(nodes: List[Dict[str, Any]]) -> bool:
    """True if any retrieved node has a non-empty page_ref."""
    return any(bool(_page_ref(n)) for n in nodes)


def _section_match(retrieved_id: str, expected_id: str) -> bool:
    """Ancestor/descendant match by dot-path prefix.

    "ch2.s1" matches "ch2" (retrieved is more specific than expected).
    "ch2" matches "ch2.s1" (retrieved is an ancestor of expected).
    """
    r, e = retrieved_id.strip(), expected_id.strip()
    if not r or not e:
        return False
    return r == e or r.startswith(e + ".") or e.startswith(r + ".")


def section_citation_f1(
    nodes: List[Dict[str, Any]],
    expected_sections: Optional[List[str]],
) -> Optional[float]:
    """F1 score between retrieved section IDs and expected_sections.

    Returns None when expected_sections is missing or empty (skip from mean).
    Returns 0.0 when nodes have no recognisable section IDs.
    """
    if not expected_sections:
        return None

    r_ids = [_node_id(n) for n in nodes]
    r_ids = [x for x in r_ids if x]
    e_ids = [str(e) for e in expected_sections if e]

    if not e_ids:
        return None
    if not r_ids:
        return 0.0

    # Precision: fraction of retrieved that hit any expected section
    p_hits = sum(1 for r in r_ids if any(_section_match(r, e) for e in e_ids))
    precision = p_hits / len(r_ids)

    # Recall: fraction of expected sections covered by any retrieved
    r_hits = sum(1 for e in e_ids if any(_section_match(r, e) for r in r_ids))
    recall = r_hits / len(e_ids)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
