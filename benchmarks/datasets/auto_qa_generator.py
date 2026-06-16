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
# PHASE 3-2: medical-domain generation
# --------------------------------------------------------------------------- #
# Index files of the medical/biomedical documents in this corpus.
MEDICAL_DOCS = [
    "c7b780a9_생체의공학개론#11_index.json",
    "92928ecd_생체의공학개론_보고서_index.json",
    "61dd7aa0_s41598-026-41649-2_reference_index.json",
]

MEDICAL_QA_PROMPT = """
You are a medical QA dataset creator for evaluating clinical document RAG.
Generate {n} questions that a clinician would realistically ask when
consulting this document.

Document tree:
{tree_json}

Question types to generate:
- clinical_fact (4): Specific clinical values, dosages, criteria
  e.g. "What is the recommended dosage of X for condition Y?"
- procedure (3): Step-by-step clinical procedures
  e.g. "What are the steps for performing X?"
- comparison (2): Comparing conditions/treatments
  e.g. "What distinguishes condition A from condition B?"
- safety (1): Contraindications, warnings, side effects
  e.g. "What are the contraindications for X?"

For each question:
- question: the clinical question
- expected_sections: relevant node IDs
- expected_answer_hint: expected answer in 1-2 sentences
- difficulty: "easy" | "medium" | "hard"
- category: one of the four types above
- clinical_relevance: why a clinician would ask this

JSON only: {{"questions": [...]}}
"""

# Default per-category counts for medical generation (sums to 14 -> 40+ over 3 docs).
_MED_COUNTS = [("clinical_fact", 6), ("procedure", 4), ("comparison", 3), ("safety", 1)]


def generate_offline_medical_questions(tree: Dict[str, Any], n: int = 14) -> List[Dict[str, Any]]:
    """Deterministic medical Q&A from a tree (clinical_fact/procedure/comparison/safety)."""
    nodes = _content_nodes(flatten_tree(tree))
    if not nodes:
        return []
    ko = _is_korean(" ".join(nd["title"] + nd["summary"] for nd in nodes))
    out: List[Dict[str, Any]] = []
    idx = 0

    def take(count):
        nonlocal idx
        chosen = []
        for _ in range(count):
            chosen.append(nodes[idx % len(nodes)])
            idx += 1
        return chosen

    for category, count in _MED_COUNTS:
        for node in take(count):
            if category == "clinical_fact":
                q = ("{0}의 핵심 임상 내용은 무엇인가?" if ko else "What is the key clinical content of {0}?").format(node["title"])
                rel = "임상 판단에 필요한 핵심 사실 확인" if ko else "Confirms a key clinical fact needed for decisions"
                diff = "easy"
            elif category == "procedure":
                q = ("{0}의 수행 절차를 단계별로 설명하시오." if ko else "What are the steps for performing {0}?").format(node["title"])
                rel = "시술/절차의 정확한 수행 순서 확인" if ko else "Verifies the correct order of a clinical procedure"
                diff = "medium"
            elif category == "comparison":
                other = nodes[(idx) % len(nodes)]
                idx_pair = other
                q = ("{0}와(과) {1}의 차이점은 무엇인가?" if ko else "What distinguishes {0} from {1}?").format(node["title"], idx_pair["title"])
                rel = "유사 조건/치료의 감별" if ko else "Differentiates similar conditions or treatments"
                diff = "hard"
                out.append({
                    "question": q,
                    "expected_sections": [node["id"], idx_pair["id"]],
                    "expected_answer_hint": ("{0} {1}".format(node["summary"], idx_pair["summary"])).strip(),
                    "difficulty": diff,
                    "category": category,
                    "clinical_relevance": rel,
                })
                continue
            else:  # safety
                q = ("{0}와 관련된 주의사항이나 금기사항은 무엇인가?" if ko else "What are the contraindications or warnings for {0}?").format(node["title"])
                rel = "환자 안전을 위한 금기/주의 확인" if ko else "Checks contraindications for patient safety"
                diff = "medium"
            out.append({
                "question": q,
                "expected_sections": [node["id"]],
                "expected_answer_hint": node["summary"],
                "difficulty": diff,
                "category": category,
                "clinical_relevance": rel,
            })
    return out[:n]


def generate_gemini_medical_questions(tree: Dict[str, Any], n: int = 14) -> List[Dict[str, Any]]:
    """Medical Q&A via Gemini (clinical prompt). Raises on API/parse failure."""
    from src.config import Config

    prompt = MEDICAL_QA_PROMPT.format(n=n, tree_json=json.dumps(tree, ensure_ascii=False))
    response = Config.CLIENT.models.generate_content(
        model=Config.MODEL_NAME, contents=prompt, config=Config.get_generation_config()
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1] if text.count("```") >= 2 else text.strip("`")
        text = text.replace("json", "", 1).strip() if text.lstrip().startswith("json") else text
    data = json.loads(text)
    raw = data.get("questions", []) if isinstance(data, dict) else data
    cleaned = []
    for item in raw:
        cleaned.append({
            "question": item.get("question", "").strip(),
            "expected_sections": list(item.get("expected_sections", []) or []),
            "expected_answer_hint": item.get("expected_answer_hint", "").strip(),
            "difficulty": item.get("difficulty", "medium"),
            "category": item.get("category", "clinical_fact"),
            "clinical_relevance": item.get("clinical_relevance", ""),
        })
    return cleaned


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def list_index_files(index_dir: Path = DEFAULT_INDEX_DIR) -> List[Path]:
    return sorted(p for p in index_dir.glob("*.json") if p.stat().st_size > 0)


def build_benchmark(
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: str = "auto",
    per_doc: int = 10,
    domain: str = "general",
    files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Generate questions across every index file and merge into one dataset.

    ``domain="medical"`` uses the clinical prompt / offline medical generator
    and the four medical categories (clinical_fact, procedure, comparison,
    safety). ``files`` overrides which index files are used.
    """
    if backend == "auto":
        backend = "gemini" if _gemini_available() else "offline"

    if domain == "medical":
        gemini_gen = generate_gemini_medical_questions
        offline_gen = generate_offline_medical_questions
    else:
        gemini_gen = generate_gemini_questions
        offline_gen = generate_offline_questions
    gen = gemini_gen if backend == "gemini" else offline_gen

    files = list(files) if files is not None else list_index_files(index_dir)
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
            print("  WARN {0} generation failed for {1}: {2}; using offline".format(backend, doc_id, exc))
            doc_questions = offline_gen(tree, n=per_doc)

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

    import unicodedata

    def _nfc(s):
        return unicodedata.normalize("NFC", s)

    available = {_nfc(p.name) for p in index_dir.glob("*.json")}
    available |= {_nfc(p.stem) for p in index_dir.glob("*.json")}
    for q in questions:
        doc = _nfc(q.get("document_id", ""))
        if doc not in available and _nfc("{0}.json".format(doc)) not in available:
            errors.append("Unresolvable document_id: {0!r} (q={1})".format(doc, q.get("question_id")))

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
    parser.add_argument("--per-doc", type=int, default=0, help="0 = domain default")
    parser.add_argument("--domain", choices=["general", "medical"], default="general")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    index_dir = Path(args.index_dir)
    medical = args.domain == "medical"
    per_doc = args.per_doc or (14 if medical else 10)
    out_path = Path(args.output)
    if medical and args.output == str(DEFAULT_OUTPUT):
        out_path = DEFAULT_OUTPUT.parent / "medical_benchmark.json"

    if args.validate_only:
        with open(out_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        result = validate_dataset(dataset, index_dir)
        return 0 if result["passed"] else 1

    files = None
    if medical:
        files = [index_dir / name for name in MEDICAL_DOCS if (index_dir / name).exists()]
    dataset = build_benchmark(index_dir, backend=args.backend, per_doc=per_doc,
                              domain=args.domain, files=files)
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
