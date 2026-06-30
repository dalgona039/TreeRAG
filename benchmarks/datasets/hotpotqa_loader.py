"""
HotpotQA multi-hop benchmark loader (PHASE 2 of the ACM upgrade plan).

``load_hotpotqa_subset`` tries to download the official HotpotQA dev set and
falls back to a bundled 20-question sample (``hotpotqa_sample.json``) when the
download is blocked (e.g. offline sandbox).

``convert_to_benchmark_format`` turns HotpotQA items into the same schema as
``full_benchmark.json`` so ``run_real_evaluation.py`` can process them. Each
item's sentence-list context is concatenated into a flat PageIndex JSON tree
(written to ``data/indices/``) so every system — including TreeRAG and RAPTOR —
can index it through the existing disk-based loaders.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = _PROJECT_ROOT / "benchmarks" / "datasets" / "hotpotqa_sample.json"
INDEX_DIR = _PROJECT_ROOT / "data" / "indices"
INDEX_PREFIX = "hotpotqa_"

# Local copy of the HotpotQA dev set (download once, then re-use offline).
# Override with env var HOTPOTQA_LOCAL=/abs/path/to/hotpot_dev_*.json
LOCAL_DEV_PATH = Path(
    os.environ.get(
        "HOTPOTQA_LOCAL",
        str(_PROJECT_ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"),
    )
)
# Remote mirrors tried in order. The original CMU host now returns HTTP 403 in
# many environments; keep it first for backward compatibility, then mirrors.
HOTPOTQA_URLS = [
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
]
# Back-compat alias (some callers import HOTPOTQA_URL).
HOTPOTQA_URL = HOTPOTQA_URLS[0]


def _normalize_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a native HotpotQA item to the spec's dict schema."""
    supporting = []
    for sf in raw.get("supporting_facts", []) or []:
        if isinstance(sf, (list, tuple)) and len(sf) >= 2:
            supporting.append({"title": sf[0], "sent_id": sf[1]})
        elif isinstance(sf, dict):
            supporting.append({"title": sf.get("title", ""), "sent_id": sf.get("sent_id", 0)})
    context = []
    for ctx in raw.get("context", []) or []:
        if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
            context.append({"title": ctx[0], "sentences": list(ctx[1])})
        elif isinstance(ctx, dict):
            context.append({"title": ctx.get("title", ""), "sentences": ctx.get("sentences", [])})
    return {
        "question_id": raw.get("_id", ""),
        "question": raw.get("question", ""),
        "answer": raw.get("answer", ""),
        "type": raw.get("type", "bridge"),
        "supporting_facts": supporting,
        "context": context,
    }


def _load_sample() -> List[Dict[str, Any]]:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_local_dev() -> List[Dict[str, Any]]:
    """Load the full HotpotQA dev set from a local file, if present."""
    if LOCAL_DEV_PATH.is_file():
        with open(LOCAL_DEV_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _download_dev() -> List[Dict[str, Any]]:
    """Try each remote mirror in turn; return [] if all fail."""
    for url in HOTPOTQA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # nosec - public dataset
                data = json.loads(resp.read().decode("utf-8"))
                if data:
                    # Cache locally for future offline runs.
                    try:
                        LOCAL_DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(LOCAL_DEV_PATH, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    except Exception:
                        pass
                    return data
        except Exception as exc:  # noqa: BLE001
            print(f"  [hotpotqa_loader] mirror failed ({url[:48]}...): {exc}")
    return []


def load_hotpotqa_subset(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Load up to ``n`` multi-hop (comparison/bridge) questions.

    Resolution order:
      1. local dev file (``data/hotpotqa/hotpot_dev_distractor_v1.json`` or
         ``$HOTPOTQA_LOCAL``) — preferred, works fully offline;
      2. download from a remote mirror (cached locally on success);
      3. bundled 20-question sample (``hotpotqa_sample.json``) as last resort.

    Filters to comparison/bridge questions with answer < 100 chars. Emits a
    clear warning when it has to fall back to the 20-question sample, so an
    intended large-scale run is never silently truncated to n = 20.
    """
    raw_items = _load_local_dev()
    source = "local file" if raw_items else None
    if not raw_items:
        raw_items = _download_dev()
        source = "remote mirror" if raw_items else None
    if not raw_items:
        raw_items = _load_sample()
        source = "bundled 20q sample"
        print(
            "  [hotpotqa_loader] WARNING: using the bundled 20-question sample. "
            "To run n>20, place the official dev set at "
            f"'{LOCAL_DEV_PATH}' or set $HOTPOTQA_LOCAL."
        )

    filtered = [
        _normalize_item(it)
        for it in raw_items
        if it.get("type") in ("comparison", "bridge")
        and len(str(it.get("answer", ""))) < 100
    ]

    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:n]
    print(
        f"  [hotpotqa_loader] source={source}; "
        f"available={len(filtered)}, requested={n}, selected={len(selected)}"
    )
    if len(selected) < n:
        print(
            f"  [hotpotqa_loader] NOTE: only {len(selected)} questions available "
            f"(< requested {n})."
        )
    return selected


def _build_flat_tree(item: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """Build a flat PageIndex JSON tree from a HotpotQA item's context."""
    support_titles = {sf["title"] for sf in item["supporting_facts"]}
    children = []
    for i, ctx in enumerate(item["context"]):
        node_id = "SEC{0}".format(i)
        children.append(
            {
                "id": node_id,
                "title": ctx["title"],
                "summary": " ".join(ctx["sentences"]).strip(),
                "page_ref": str(i + 1),
            }
        )
    return {
        "id": "ROOT",
        "title": item["question"][:60],
        "summary": "HotpotQA multi-hop context for: {0}".format(item["question"]),
        "page_ref": "1-{0}".format(max(1, len(children))),
        "children": children,
    }


def convert_to_benchmark_format(
    hotpotqa_items: List[Dict[str, Any]], write_indices: bool = True
) -> Dict[str, Any]:
    """Convert HotpotQA items to full_benchmark.json schema.

    Writes one flat PageIndex JSON per item into ``data/indices/`` (prefixed
    ``hotpotqa_``) so the existing runner can load them by document_id.
    """
    if write_indices:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    questions = []
    documents = []
    for item in hotpotqa_items:
        qid = item["question_id"] or "hp_{0}".format(len(questions))
        doc_id = "{0}{1}_index.json".format(INDEX_PREFIX, qid)
        tree = _build_flat_tree(item, doc_id)

        if write_indices:
            with open(INDEX_DIR / doc_id, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)

        support_titles = {sf["title"] for sf in item["supporting_facts"]}
        expected_sections = [c["id"] for c in tree["children"] if c["title"] in support_titles]
        difficulty = "hard" if item["type"] == "bridge" else "medium"
        questions.append(
            {
                "question_id": qid,
                "document_id": doc_id,
                "question": item["question"],
                "expected_sections": expected_sections or [c["id"] for c in tree["children"]],
                "expected_answer_hint": item["answer"],
                "difficulty": difficulty,
                "category": "multi_hop",
                "hotpotqa_type": item["type"],
            }
        )
        documents.append(doc_id)

    return {
        "version": "2.0-hotpotqa",
        "backend": "hotpotqa",
        "total_questions": len(questions),
        "documents": documents,
        "questions": questions,
    }


def build_hotpotqa_dataset(n: int = 100, seed: int = 42) -> Dict[str, Any]:
    return convert_to_benchmark_format(load_hotpotqa_subset(n=n, seed=seed))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HotpotQA benchmark (PHASE 2)")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "benchmarks/datasets/hotpotqa_benchmark.json"))
    args = parser.parse_args()

    ds = build_hotpotqa_dataset(n=args.n, seed=args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)
    print("HotpotQA benchmark: {0} questions -> {1}".format(ds["total_questions"], args.output))
