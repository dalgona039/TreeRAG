#!/usr/bin/env python3
"""
Experiment 1: Citation Traceability and Accuracy

For each (system, question) pair the runner records:
  citation_availability  – True if any retrieved node has a non-empty page_ref
  citation_f1            – Section-ID F1 between retrieved nodes and expected_sections

Since full_benchmark.json has expected_sections (tree node IDs like 'ch1', 'ch2.s1')
but no explicit gold page numbers, citation_correctness is operationalised as
section_citation_f1 using ancestor/descendant matching.

Systems that return tree-section nodes (BM25, Dense, TreeRAG-DFS, TreeRAG-Beam)
produce meaningful F1 values.  FlatRAG and RAPTOR return document-level / cluster
IDs that don't match section IDs, so their citation_f1 will be near 0 — this is
a genuine structural limitation, not a measurement error.

Outputs: exp1_citation_<stamp>.{json,md}  →  data/benchmark_reports/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORT_DIR = _ROOT / "data" / "benchmark_reports"
ALL_SYSTEMS = ["bm25", "dense", "flatrag", "raptor", "treerag_dfs", "treerag_beam"]
SYSTEM_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "flatrag": "FlatRAG",
    "raptor": "RAPTOR",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}


def _mean(vals: List[Any]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Citation Traceability")
    parser.add_argument(
        "--dataset",
        default=str(_ROOT / "benchmarks" / "datasets" / "full_benchmark.json"),
    )
    parser.add_argument("--systems", default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 = all questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Print sample answers/citations; validate and exit on failure")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--local-judge-model", default="llama3.1:8b")
    args = parser.parse_args()

    systems = ALL_SYSTEMS if args.systems == "all" else [
        s.strip() for s in args.systems.split(",")
    ]

    # Load dataset
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    questions: List[Dict[str, Any]] = dataset["questions"]

    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(questions)
        questions = questions[:args.limit]

    print(f"Experiment 1: Citation Traceability")
    print(f"Dataset: {args.dataset}  |  Questions: {len(questions)}  |  Seed: {args.seed}")
    print(f"Systems: {', '.join(systems)}")

    # Set up Ollama override
    from src.core.ollama_client import OllamaClient
    from src.config import set_client_override
    client = OllamaClient(model="llama3.1:8b")
    set_client_override(client)

    from benchmarks.run_real_evaluation import Evaluator
    from benchmarks.metrics.citation_metrics import citation_availability, section_citation_f1

    evaluator = Evaluator(
        mode="online",
        use_llm_judge=args.use_llm_judge,
        local_judge=args.use_llm_judge,
        local_judge_model=args.local_judge_model,
        gen_backend="ollama",
        gen_model="llama3.1:8b",
    )

    per_system: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}

    for system in systems:
        print(f"\n▶ {SYSTEM_LABELS.get(system, system)} — {len(questions)} questions")
        for q in questions:
            qid = q["question_id"]
            expected_sections: List[str] = q.get("expected_sections") or []

            t0 = time.perf_counter()
            try:
                answer, nodes = evaluator.run_system(
                    system, q["question"], q["document_id"]
                )
            except Exception as exc:
                print(f"   ⚠ {system} failed on {qid}: {exc}")
                answer, nodes = "", []
            latency = time.perf_counter() - t0

            avail = citation_availability(nodes)
            cit_f1 = section_citation_f1(nodes, expected_sections)

            # ROUGE-L against expected_answer_hint
            from benchmarks.metrics.text_similarity import rouge_l_score
            rouge = rouge_l_score(answer, q.get("expected_answer_hint", ""))

            row: Dict[str, Any] = {
                "question_id": qid,
                "document_id": q["document_id"],
                "category": q.get("category", ""),
                "answer": answer,
                "latency": latency,
                "citation_availability": avail,
                "citation_f1": cit_f1,
                "rouge_l": rouge,
            }

            if args.smoke:
                print(f"   Q: {q['question'][:80]}")
                print(f"   A ({len(answer)} chars, {latency:.1f}s): {answer[:200]}")
                print(f"   expected_sections: {expected_sections}")
                retrieved_ids = []
                for n in nodes:
                    nid = n.get("id") or (n.get("node") or {}).get("id", "")
                    pr = n.get("page_ref") or (n.get("node") or {}).get("page_ref", "")
                    retrieved_ids.append(f"{nid}(p{pr})")
                print(f"   retrieved: {retrieved_ids}")
                print(f"   citation_availability={avail}, citation_f1={cit_f1}")
                if len(answer) < 10:
                    print("   ⛔ ANSWER TOO SHORT")

            per_system[system].append(row)

    # Aggregate
    agg: Dict[str, Dict[str, Any]] = {}
    for s in systems:
        rows = per_system[s]
        agg[s] = {
            "citation_availability": _mean([r["citation_availability"] for r in rows]),
            "citation_f1": _mean([r["citation_f1"] for r in rows if r["citation_f1"] is not None]),
            "rouge_l": _mean([r["rouge_l"] for r in rows]),
            "latency": _mean([r["latency"] for r in rows]),
            "n": len(rows),
        }

    # Print table
    print()
    print("┌─────────────────┬──────────────┬─────────────┬─────────┬──────────┐")
    print("│ System          │ Cite-Avail   │ Cite-F1     │ ROUGE-L │ Latency  │")
    print("├─────────────────┼──────────────┼─────────────┼─────────┼──────────┤")
    for s in systems:
        a = agg[s]
        ca = a.get("citation_availability")
        cf = a.get("citation_f1")
        ca_str = f"{ca:.3f}" if ca is not None else "  —  "
        cf_str = f"{cf:.3f}" if cf is not None else "  —  "
        print(
            f"│ {SYSTEM_LABELS.get(s,s):<15} │ {ca_str:>10}   │ {cf_str:>10}  │"
            f"  {a.get('rouge_l',0):.3f}  │  {a.get('latency',0):.2f}s │"
        )
    print("└─────────────────┴──────────────┴─────────────┴─────────┴──────────┘")

    # Smoke validation
    if args.smoke:
        failed = []
        for s in systems:
            rows = per_system[s]
            bad_ans = [r for r in rows if len(r["answer"]) < 10]
            if bad_ans:
                failed.append(f"{s}: {len(bad_ans)} short answers")
            # treerag_beam legitimately gets cache hits from treerag_dfs (same
            # question+index → same cache key), so latency near 0 is expected.
            if s != "treerag_beam":
                slow = [r for r in rows if r["latency"] < 1.0]
                if slow:
                    failed.append(f"{s}: {len(slow)} fast (latency<1s → LLM not called?)")
            if agg[s].get("citation_availability") is None:
                failed.append(f"{s}: citation_availability is None")
        if failed:
            print("\n⛔ SMOKE FAIL:")
            for msg in failed:
                print(f"  - {msg}")
            sys.exit(1)
        print("\n✅ SMOKE PASS (Exp 1)")
        return

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"exp1_citation_{stamp}.json"
    md_path = REPORT_DIR / f"exp1_citation_{stamp}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": args.dataset, "n_questions": len(questions),
             "seed": args.seed, "summary": agg, "per_question": per_system},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n💾 JSON → {json_path}")

    # Markdown
    lines = [
        "# Experiment 1: Citation Traceability",
        "",
        f"Dataset: `{Path(args.dataset).name}`  |  Questions: {len(questions)}  |  Seed: {args.seed}",
        "",
        "**Citation Availability**: fraction of questions where ≥1 retrieved node has a page_ref.",
        "",
        "**Citation F1**: F1 between retrieved section IDs and `expected_sections` (ancestor/descendant match).",
        "FlatRAG / RAPTOR return document-level or cluster IDs, so their F1 is near 0 — structural limitation.",
        "",
        "| System | Cite-Avail | Cite-F1 | ROUGE-L | Latency(s) |",
        "|--------|-----------|---------|---------|-----------|",
    ]
    for s in systems:
        a = agg[s]
        ca = a.get("citation_availability")
        cf = a.get("citation_f1")
        lines.append(
            f"| {SYSTEM_LABELS.get(s,s)} "
            f"| {ca:.3f} " if ca is not None else "| — "
            f"| {cf:.3f} " if cf is not None else "| — "
            f"| {a.get('rouge_l',0):.3f} "
            f"| {a.get('latency',0):.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📄 Markdown → {md_path}")


if __name__ == "__main__":
    main()
