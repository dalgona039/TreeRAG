"""GovReport long-document benchmark loader (P1-D).

Builds a TreeRAG-compatible benchmark from the GovReport summarization dataset
(ccdv/govreport-summarization on HuggingFace). Each ~10k-word government report
is chunked into 400-word page-segments and stored as a flat PageIndex tree,
compatible with all six retrieval systems.

Question: "What are the main findings, conclusions, and recommendations of
this report?" (comprehension question designed to require broad retrieval).
Expected answer: the provided abstractive summary.

Resolution order for the dataset:
  1. Local JSON cache at data/govreport/govreport_val.json
  2. HuggingFace datasets library (requires network on first run)

Usage:
  # Build and cache locally (one-time):
  python benchmarks/datasets/govreport_loader.py -n 40
  # Then run evaluation:
  python benchmarks/run_real_evaluation.py \
      --dataset benchmarks/datasets/govreport_benchmark.json \
      --systems all --limit 40 --seed 42 \
      --gen-backend ollama --gen-model llama3.1:8b \
      --use-llm-judge --local-judge --local-judge-model llama3.1:8b \
      --output data/benchmark_reports/online_local_llama_govreport.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE = _PROJECT_ROOT / "data" / "govreport" / "govreport_val.json"
INDEX_DIR = _PROJECT_ROOT / "data" / "indices"
INDEX_PREFIX = "govreport_"
CHUNK_WORDS = 400  # ~400 words per page segment


def _load_raw() -> List[Dict[str, str]]:
    """Load raw GovReport items, preferring local cache."""
    if LOCAL_CACHE.is_file():
        with open(LOCAL_CACHE, encoding="utf-8") as f:
            return json.load(f)
    print("[govreport_loader] Downloading ccdv/govreport-summarization from HuggingFace…")
    from datasets import load_dataset
    ds = load_dataset("ccdv/govreport-summarization", split="validation")
    items = [{"report": r["report"], "summary": r["summary"]} for r in ds]
    LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"[govreport_loader] Cached {len(items)} items → {LOCAL_CACHE}")
    return items


def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> List[str]:
    """Split text into chunks of ~chunk_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[i:i + chunk_words]))
    return chunks or [""]


def _build_index_tree(report_text: str, doc_id: str) -> Dict[str, Any]:
    """Build a flat PageIndex tree from a government report."""
    chunks = _chunk_text(report_text)
    children = []
    for i, chunk in enumerate(chunks):
        children.append({
            "id": f"PAGE{i + 1}",
            "title": f"Section {i + 1}",
            "summary": chunk,
            "page_ref": str(i + 1),
        })
    total_words = len(report_text.split())
    return {
        "id": "ROOT",
        "title": doc_id.replace(INDEX_PREFIX, "").replace("_index.json", ""),
        "summary": f"Government report ({total_words} words, {len(chunks)} sections)",
        "page_ref": f"1-{len(chunks)}",
        "children": children,
    }


QUESTION_TEMPLATE = (
    "What are the main findings, conclusions, and recommendations of this report?"
)


def build_govreport_dataset(n: int = 40, seed: int = 42,
                            write_indices: bool = True) -> Dict[str, Any]:
    """Build a benchmark dict compatible with run_real_evaluation.py."""
    import random
    raw = _load_raw()
    rng = random.Random(seed)
    rng.shuffle(raw)
    selected = raw[:n]
    print(f"[govreport_loader] {len(selected)} reports selected (n={n}, seed={seed})")

    if write_indices:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    questions = []
    documents = []
    for i, item in enumerate(selected):
        doc_id = f"{INDEX_PREFIX}{i:04d}_index.json"
        tree = _build_index_tree(item["report"], doc_id)

        if write_indices:
            with open(INDEX_DIR / doc_id, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)

        n_pages = len(tree["children"])
        questions.append({
            "question_id": f"gr_{i:04d}",
            "document_id": doc_id,
            "question": QUESTION_TEMPLATE,
            "expected_sections": [c["id"] for c in tree["children"]],
            "expected_answer_hint": item["summary"],
            "difficulty": "hard",
            "category": "long_document",
            "n_pages": n_pages,
            "report_words": len(item["report"].split()),
        })
        documents.append(doc_id)

    return {
        "version": "1.0-govreport",
        "backend": "govreport",
        "total_questions": len(questions),
        "documents": documents,
        "questions": questions,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GovReport benchmark (P1-D)")
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "benchmarks" / "datasets" / "govreport_benchmark.json"),
    )
    args = parser.parse_args()

    ds = build_govreport_dataset(n=args.n, seed=args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)
    print(f"GovReport benchmark: {ds['total_questions']} questions → {args.output}")
    avg_words = sum(q["report_words"] for q in ds["questions"]) / len(ds["questions"])
    avg_pages = sum(q["n_pages"] for q in ds["questions"]) / len(ds["questions"])
    print(f"Avg report: {avg_words:.0f} words, {avg_pages:.0f} page-segments")
