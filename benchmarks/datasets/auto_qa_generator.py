"""
Auto Q&A Generator (PHASE A-1 / A-2 of the KCI publication plan).

Generates a RAG evaluation benchmark from PageIndex JSON trees in
``data/indices/``.

Two generation backends are supported:

* ``gemini``  – calls the Gemini model via :class:`src.config.Config` using the
  prompt template from the KCI plan. Produces diverse factual / multi-hop /
  comparative questions. Requires network access to the Gemini API.
* ``offline`` – a deterministic, network-free fallback that derives questions
  directly from the tree structure (real node ids as ground-truth sections).
  Used automatically when the Gemini API is unreachable so the rest of the
  evaluation pipeline can still run end-to-end.

CLI::

    python benchmarks/datasets/auto_qa_generator.py            # auto backend
    python benchmarks/datasets/auto_qa_generator.py --backend offline
    python benchmarks/datasets/auto_qa_generator.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``src`` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATASET_VERSION = "2.0"
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "data" / "indices"
DEFAULT_OUTPUT = _PROJECT_ROOT / "benchmarks" / "datasets" / "full_benchmark.json"

REQUIRED_FIELDS = (
    "question_id",
    "document_id",
    "question",
    "expected_sections",
    "expected_answer_hint",
    "difficulty",
    "category",
)

GEMINI_PROMPT_TEMPLATE = """You are a QA dataset creator for a RAG evaluation benchmark.
Given the following document tree structure, generate {n} diverse questions
that test different retrieval scenarios.

Document tree:
{tree_json}

Generate questions in three categories:
- factual (5): Single-section lookup, e.g. "What is X?"
- multi_hop (3): Requires combining 2+ sections, e.g. "How does A relate to B?"
- comparative (2): Requires comparison, e.g. "What are the differences between X and Y?"

For each question provide:
- question: the question string
- expected_sections: list of node IDs that contain the answer
- expected_answer_hint: a brief expected answer (1-2 sentences)
- difficulty: "easy" | "medium" | "hard"
- category: "factual" | "multi_hop" | "comparative"

Respond in JSON only:
{{"questions": [...]}}"""


# --------------------------------------------------------------------------- #
# Tree helpers
# --------------------------------------------------------------------------- #
def flatten_tree(node: Dict[str, Any], acc: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return a flat list of nodes (id/title/summary/page_ref/depth)."""
    if acc is None:
        acc = []
    if not isinstance(node, dict):
        return acc
    acc.append(
        {
            "id": node.get("id", f"node_{len(acc)}"),
            "title": node.get("title", ""),
            "summary": node.get("summary", ""),
            "page_ref": node.get("page_ref", ""),
        }
    )
    for child in node.get("children", []) or []:
        flatten_tree(child, acc)
    return acc


def _is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def _content_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nodes that carry a usable title + summary (skip the ROOT wrapper)."""
    out = []
    for n in nodes:
        if n["id"] == "ROOT":
            continue
        if n["title"] and n["summary"]:
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# Offline (deterministic) generation
# --------------------------------------------------------------------------- #
def generate_offline_questions(tree: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    """Derive ``n`` questions deterministically from a single tree.

    Layout mirrors the Gemini spec: 5 factual, 3 multi_hop, 2 comparative.
    Ground-truth ``expected_sections`` use real node ids, and
    ``expected_answer_hint`` reuses the node summary, which makes the dataset
    directly usable for retrieval evaluation even without an LLM.
    """
    nodes = _content_nodes(flatten_tree(tree))
    if not nodes:
        return []

    ko = _is_korean(" ".join(n["title"] + n["summary"] for n in nodes))
    questions: List[Dict[str, Any]] = []

    # --- factual (5): single-section lookup --------------------------------
    for node in nodes[:5]:
        if ko:
            q = f"{node['title']}에 대해 설명하시오."
        else:
            q = f"What is {node['title']}?"
        questions.append(
            {
                "question": q,
                "expected_sections": [node["id"]],
                "expected_answer_hint": node["summary"],
                "difficulty": "easy",
                "category": "factual",
            }
        )

    # --- multi_hop (3): combine two sections -------------------------------
    pairs = []
    for i in range(len(nodes) - 1):
        pairs.append((nodes[i], nodes[i + 1]))
        if len(pairs) >= 3:
            break
    for a, b in pairs[:3]:
        if ko:
            q = f"{a['title']}와(과) {b['title']}은(는) 어떻게 연관되는가?"
        else:
            q = f"How does {a['title']} relate to {b['title']}?"
        questions.append(
            {
                "question": q,
                "expected_sections": [a["id"], b["id"]],
                "expected_answer_hint": f"{a['summary']} {b['summary']}".strip(),
                "difficulty": "hard",
                "category": "multi_hop",
            }
        )

    # --- comparative (2): compare two sections -----------------------------
    comp_pairs = []
    if len(nodes) >= 4:
        comp_pairs = [(nodes[0], nodes[2]), (nodes[1], nodes[3])]
    elif len(nodes) >= 2:
        comp_pairs = [(nodes[0], nodes[1])]
    for a, b in comp_pairs[:2]:
        if ko:
            q = f"{a['title']}와(과) {b['title']}의 차이점은 무엇인가?"
        else:
            q = f"What are the differences between {a['title']} and {b['title']}?"
        questions.append(
            {
                "question": q,
                "expected_sections": [a["id"], b["id"]],
                "expected_answer_hint": f"{a['summary']} {b['summary']}".strip(),
                "difficulty": "medium",
                "category": "comparative",
            }
        )

    return questions[:n]


# --------------------------------------------------------------------------- #
# Gemini generation
# --------------------------------------------------------------------------- #
def generate_gemini_questions(tree: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    """Generate questions with Gemini. Raises on any API/parse failure."""
    from src.config import Config  # local import: requires GOOGLE_API_KEY

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        n=n, tree_json=json.dumps(tree, ensure_ascii=False)
    )
    response = Config.CLIENT.models.generate_content(
        model=Config.MODEL_NAME,
        contents=prompt,
        config=Config.get_generation_config(),
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1] if text.count("```") >= 2 else text.strip("`")
        text = text.replace("json", "", 1).strip() if text.lstrip().startswith("json") else text
    data = json.loads(text)
    raw = data.get("questions", []) if isinstance(data, dict) else data
    cleaned = []
    for item in raw:
        cleaned.append(
            {
                "question": item.get("question", "").strip(),
                "expected_sections": list(item.get("expected_sections", []) or []),
                "expected_answer_hint": item.get("expected_answer_hint", "").strip(),
                "difficulty": item.get("difficulty", "medium"),
                "category": item.get("category", "factual"),
            }
        )
    return cleaned


def _gemini_available() -> bool:
    try:
        from src.config import Config

        Config.CLIENT.models.generate_content(
            model=Config.MODEL_NAME, contents="ping"
        )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def list_index_files(index_dir: Path = DEFAULT_INDEX_DIR) -> List[Path]:
    return sorted(p for p in index_dir.glob("*.json") if p.stat().st_size > 0)


def build_benchmark(
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: str = "auto",
    per_doc: int = 10,
) -> Dict[str, Any]:
    """Generate questions across every index file and merge into one dataset."""
    if backend == "auto":
        backend = "gemini" if _gemini_available() else "offline"
    gen = generate_gemini_questions if backend == "gemini" else generate_offline_questions

    files = list_index_files(index_dir)
    documents: List[str] = []
    questions: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    counter = 1

    for path in files:
        doc_id = path.name  # full index filename → trivially resolvable
        with open(path, "r", encoding="utf-8") as f:
            tree = json.load(f)
        try:
            doc_questions = gen(tree, n=per_doc)
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"  ⚠️  {backend} generation failed for {doc_id}: {exc}; using offline")
            doc_questions = generate_offline_questions(tree, n=per_doc)

        cats: Dict[str, int] = {}
        for q in doc_questions:
            q["question_id"] = f"auto_{counter:03d}"
            q["document_id"] = doc_id
            counter += 1
            questions.append(q)
            cats[q["category"]] = cats.get(q["category"], 0) + 1

        documents.append(doc_id)
        summary.append({"document": doc_id, "count": len(doc_questions), "categories": cats})

    dataset = {
        "version": DATASET_VERSION,
        "backend": backend,
        "total_questions": len(questions),
        "documents": documents,
        "questions": questions,
    }
    _print_summary(summary, len(questions), backend)
    return dataset


def _print_summary(summary: List[Dict[str, Any]], total: int, backend: str) -> None:
    print(f"\n=== Q&A Generation Summary (backend={backend}) ===")
    print(f"{'Document':<48}{'Count':>7}  Categories")
    print("-" * 90)
    for row in summary:
        cats = ", ".join(f"{k}:{v}" for k, v in sorted(row["categories"].items()))
        name = row["document"][:46]
        print(f"{name:<48}{row['count']:>7}  {cats}")
    print("-" * 90)
    print(f"{'TOTAL':<48}{total:>7}")


# --------------------------------------------------------------------------- #
# PHASE A-2: validation
# --------------------------------------------------------------------------- #
def validate_dataset(
    dataset: Dict[str, Any], index_dir: Path = DEFAULT_INDEX_DIR
) -> Dict[str, Any]:
    """Check for duplicate questions, resolvable doc ids, and required fields."""
    errors: List[str] = []
    questions = dataset.get("questions", [])

    # A question is a duplicate only if the same text targets the same
    # document. The same question may legitimately be asked of different
    # documents in a multi-document benchmark.
    seen = set()
    for q in questions:
        key = (q.get("document_id", ""), q.get("question", "").strip())
        if key in seen:
            errors.append(f"Duplicate question in {key[0]}: {key[1]!r}")
        seen.add(key)

    available = {p.name for p in index_dir.glob("*.json")}
    available |= {p.stem for p in index_dir.glob("*.json")}
    for q in questions:
        doc = q.get("document_id", "")
        if doc not in available and f"{doc}.json" not in available:
            errors.append(f"Unresolvable document_id: {doc!r} (q={q.get('question_id')})")

    for q in questions:
        for field in REQUIRED_FIELDS:
            if field not in q or q[field] in (None, "", []):
                errors.append(f"Missing/empty field {field!r} in {q.get('question_id')}")

    passed = not errors
    print("\n=== Dataset Validation (PHASE A-2) ===")
    print(f"Total questions : {len(questions)}")
    print(f"Unique questions: {len(seen)}")
    print(f"Documents       : {len(dataset.get('documents', []))}")
    if passed:
        print("RESULT: PASS ✅")
    else:
        print(f"RESULT: FAIL ❌ ({len(errors)} issue(s))")
        for e in errors[:20]:
            print(f"  - {e}")
    return {"passed": passed, "errors": errors}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Auto Q&A generator (PHASE A)")
    parser.add_argument("--backend", choices=["auto", "gemini", "offline"], default="auto")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-doc", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    index_dir = Path(args.index_dir)
    out_path = Path(args.output)

    if args.validate_only:
        with open(out_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        result = validate_dataset(dataset, index_dir)
        return 0 if result["passed"] else 1

    dataset = build_benchmark(index_dir, backend=args.backend, per_doc=args.per_doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Wrote {dataset['total_questions']} questions → {out_path}")

    result = validate_dataset(dataset, index_dir)
    if dataset["total_questions"] < 50:
        print(f"⚠️  Only {dataset['total_questions']} questions (<50 target).")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
