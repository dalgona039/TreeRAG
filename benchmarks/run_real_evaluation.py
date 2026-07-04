#!/usr/bin/env python3
"""
End-to-end evaluation runner (PHASE C-3 of the KCI publication plan).

Runs every benchmark question through one or more retrieval systems, scores the
answers with ROUGE-L / BERTScore / (optional) LLM-judge, runs paired
significance tests of TreeRAG-Beam against each baseline, and writes a JSON
report plus a comparison table.

Systems: ``bm25``, ``dense``, ``flatrag``, ``treerag_dfs``, ``treerag_beam``
(or ``all``).

Online vs offline:
  The runner pings Gemini once. When reachable it uses the real
  :class:`TreeRAGReasoner` for the TreeRAG systems and :class:`GeminiJudge` for
  the LLM-judge axis. When unreachable (e.g. sandbox without network) it falls
  back to deterministic keyword traversal + extractive answers and a BERTScore
  proxy, so the full pipeline still runs. Use ``--mode`` to force a mode.

Usage::

    python benchmarks/run_real_evaluation.py \
        --dataset benchmarks/datasets/full_benchmark.json --systems all
    python benchmarks/run_real_evaluation.py --systems bm25,treerag_beam --use-llm-judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.metrics.text_similarity import bertscore_f1, rouge_l_score, _token_f1
from benchmarks.metrics.statistical_tests import StatisticalTests
from src.config import Config

ALL_SYSTEMS = ["bm25", "dense", "flatrag", "raptor", "treerag_dfs", "treerag_beam"]
SYSTEM_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "flatrag": "FlatRAG",
    "raptor": "RAPTOR",
    "treerag_dfs": "TreeRAG-DFS",
    "treerag_beam": "TreeRAG-Beam",
}
DEFAULT_REPORT_DIR = _PROJECT_ROOT / "data" / "benchmark_reports"
RAW_TEXT_DIR = _PROJECT_ROOT / "data" / "raw_text"


# --------------------------------------------------------------------------- #
# Tree / answer helpers
# --------------------------------------------------------------------------- #
def _flatten(node: Dict[str, Any], depth: int, acc: List[Tuple[Dict[str, Any], int]]) -> None:
    if not isinstance(node, dict):
        return
    acc.append((node, depth))
    for child in node.get("children", []) or []:
        _flatten(child, depth + 1, acc)


def _node_text(node: Dict[str, Any]) -> str:
    return f"{node.get('title', '')} {node.get('summary', '')}".strip()


def extractive_answer(nodes: List[Dict[str, Any]]) -> str:
    """Synthesise a deterministic answer from retrieved nodes."""
    if not nodes:
        return ""
    parts = []
    for n in nodes:
        title = n.get("title", "")
        summary = n.get("summary", "")
        page_ref = n.get("page_ref", "")
        entry = f"{title}: {summary}".strip(": ").strip()
        if page_ref:
            entry += f" [p.{page_ref}]"
        parts.append(entry)
    return " ".join(parts)


def nodes_have_page_citation(nodes: List[Dict[str, Any]]) -> bool:
    """Return True if any retrieved node carries a page reference."""
    return any(bool(n.get("page_ref", "")) for n in nodes)


def keyword_traversal(
    tree: Dict[str, Any], query: str, k: int, prefer_shallow: bool = False
) -> List[Dict[str, Any]]:
    """Keyword-scored node selection (offline DFS/Beam approximation).

    Scores every node by token-F1 against the query; ``prefer_shallow`` adds a
    mild depth penalty (DFS-style path preference) vs the wider Beam variant.
    """
    flat: List[Tuple[Dict[str, Any], int]] = []
    _flatten(tree, 0, flat)
    scored = []
    for node, depth in flat:
        if node.get("id") == "ROOT":
            continue
        s = _token_f1(query, _node_text(node))
        if prefer_shallow:
            s *= (0.9 ** depth)
        scored.append((s, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for s, n in scored[:k] if s > 0]


# --------------------------------------------------------------------------- #
# System adapters
# --------------------------------------------------------------------------- #
class Evaluator:
    def __init__(self, mode: str, use_llm_judge: bool, domain: str = "general",
                 local_judge: bool = False, local_judge_model: str = "llama3.1:8b",
                 gen_backend: str = "gemini", gen_model: str = "gemma4:12b"):
        self.mode = mode  # "online" | "offline"
        self.domain = domain
        self.gen_backend = gen_backend
        self.gen_model = gen_model
        self.use_llm_judge = use_llm_judge and (mode == "online" or local_judge)
        self._index_cache: Dict[str, Any] = {}
        self._bm25_cache: Dict[str, Any] = {}
        self._dense_cache: Dict[str, Any] = {}
        self._reasoner_cache: Dict[Tuple[str, str], Any] = {}
        self._raptor_cache: Dict[str, Any] = {}
        self._judge = None
        if self.use_llm_judge:
            if local_judge:
                from benchmarks.metrics.llm_judge import LocalJudge
                self._judge = LocalJudge(model=local_judge_model)
            else:
                from benchmarks.metrics.llm_judge import GeminiJudge
                self._judge = GeminiJudge()

    def load_tree(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self._index_cache:
            path = os.path.join(Config.INDEX_DIR, doc_id)
            if not os.path.exists(path) and not doc_id.endswith(".json"):
                path = os.path.join(Config.INDEX_DIR, f"{doc_id}.json")
            with open(path, "r", encoding="utf-8") as f:
                self._index_cache[doc_id] = json.load(f)
        return self._index_cache[doc_id]

    # -- shared LLM generation (used by all systems when gen_backend=ollama) ---
    def _llm_generate(self, context: str, question: str) -> str:
        """Generate an answer from context+question via the active LLM client.

        Used to give every system the same generation backend so comparisons are
        fair. Falls back to extractive context only if the LLM call fails.

        Prompt skeleton language matches the question language so the local 8B
        model does not drift to Korean when answering English benchmarks.
        """
        import re as _re
        from src.config import Config as _Config

        is_korean = bool(_re.search(r"[가-힣]", question))
        if is_korean:
            prompt = (
                f"아래 컨텍스트를 참고하여 질문에 한국어로 답하세요. "
                f"컨텍스트에 없는 내용은 추측하지 마세요.\n\n"
                f"### 컨텍스트:\n{context}\n\n"
                f"### 질문:\n{question}\n\n"
                f"### 답변:"
            )
        else:
            prompt = (
                f"Answer the question using ONLY the context below. "
                f"Do NOT speculate beyond the context. "
                f"Respond in English.\n\n"
                f"### Context:\n{context}\n\n"
                f"### Question:\n{question}\n\n"
                f"### Answer:"
            )
        try:
            client = _Config.get_client()  # returns override when set
            resp = client.models.generate_content(
                model=None,   # OllamaClient ignores this
                contents=prompt,
                config=None,  # no JSON format: we want natural language
            )
            answer = (resp.text or "").strip()
            if len(answer) < 10:
                print(f"   ⚠️ LLM returned very short answer ({len(answer)} chars) — extractive fallback")
                return context
            return answer
        except Exception as exc:
            print(f"   ⚠️ LLM generation failed: {exc} — extractive fallback")
            return context

    # -- individual systems -------------------------------------------------
    def _run_bm25(self, q: str, doc_id: str, branches: int):
        from src.core.bm25_baseline import BM25Retriever

        if doc_id not in self._bm25_cache:
            self._bm25_cache[doc_id] = BM25Retriever(self.load_tree(doc_id))
        nodes = self._bm25_cache[doc_id].retrieve(q, top_k=branches)
        context = extractive_answer(nodes)
        if self.gen_backend == "ollama":
            answer = self._llm_generate(context, q)
        else:
            answer = context
        return answer, nodes

    def _run_dense(self, q: str, doc_id: str, branches: int):
        from src.core.dense_retrieval_baseline import DenseRetriever

        if doc_id not in self._dense_cache:
            self._dense_cache[doc_id] = DenseRetriever(self.load_tree(doc_id))
        nodes = self._dense_cache[doc_id].retrieve(q, top_k=branches)
        context = extractive_answer(nodes)
        if self.gen_backend == "ollama":
            answer = self._llm_generate(context, q)
        else:
            answer = context
        return answer, nodes

    def _run_flatrag(self, q: str, doc_id: str, branches: int):
        from src.core.flat_rag_baseline import FlatRAGBaseline

        key = f"flat::{doc_id}"
        if key not in self._index_cache:
            self._index_cache[key] = FlatRAGBaseline([doc_id])
        baseline = self._index_cache[key]
        extractive, meta = baseline.query(q, max_branches=branches)
        nodes = [{"id": d} for d in meta.get("retrieved_docs", [])]
        if self.gen_backend == "ollama":
            answer = self._llm_generate(extractive, q)
        else:
            answer = extractive
        return answer, nodes

    def load_raw_text(self, doc_id: str) -> str:
        """Resolve a doc_id (index filename) to its extracted plain text."""
        stem = doc_id.replace("_index.json", "").replace(".json", "")
        path = RAW_TEXT_DIR / (stem + ".txt")
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        # Fallback: synthesise pseudo-text from the index tree summaries so
        # RAPTOR still has input even when raw text was not extracted.
        tree = self.load_tree(doc_id)
        flat: List[Tuple[Dict[str, Any], int]] = []

        def _walk(node, depth):
            if isinstance(node, dict):
                flat.append((node, depth))
                for c in node.get("children", []) or []:
                    _walk(c, depth + 1)

        _walk(tree, 0)
        lines = []
        for node, _ in flat:
            page = node.get("page_ref", "")
            if page:
                lines.append("--- PAGE {0} ---".format(str(page).split("-")[0]))
            lines.append("{0} {1}".format(node.get("title", ""), node.get("summary", "")).strip())
        return "\n".join(lines)

    def _run_raptor(self, q: str, doc_id: str, branches: int):
        from src.core.raptor_baseline import RaptorBaseline

        if doc_id not in self._raptor_cache:
            self._raptor_cache[doc_id] = RaptorBaseline(
                self.load_raw_text(doc_id), doc_id
            )
        result = self._raptor_cache[doc_id].answer(q, top_k=max(branches, 5))
        extractive = result["answer"]
        if self.gen_backend == "ollama":
            answer = self._llm_generate(extractive, q)
        else:
            answer = extractive
        return answer, result["source_nodes"]

    def _run_treerag(self, q: str, doc_id: str, algo: str, branches: int):
        if self.mode == "online":
            from src.core.reasoner import TreeRAGReasoner

            # Use a simple prompt for local LLM backends to avoid JSON-scaffold
            # truncation that corrupts answers with small models.
            use_simple = (self.gen_backend == "ollama")

            key = (doc_id, algo)
            if key not in self._reasoner_cache:
                self._reasoner_cache[key] = TreeRAGReasoner(
                    [doc_id],
                    traversal_algorithm="dfs" if algo == "dfs" else "beam_search",
                    enable_compression=True,
                )
            answer, meta = self._reasoner_cache[key].query(
                q, max_branches=branches, use_simple_prompt=use_simple
            )
            nodes = meta.get("nodes_selected", []) or []
            nodes = [n if isinstance(n, dict) else {"id": n} for n in nodes]
            return answer, nodes
        # offline keyword approximation
        k = branches if algo == "dfs" else max(branches, 5)
        nodes = keyword_traversal(
            self.load_tree(doc_id), q, k, prefer_shallow=(algo == "dfs")
        )
        return extractive_answer(nodes), nodes

    def run_system(self, system: str, q: str, doc_id: str, branches: int = 3):
        if system == "bm25":
            return self._run_bm25(q, doc_id, branches)
        if system == "dense":
            return self._run_dense(q, doc_id, branches)
        if system == "flatrag":
            return self._run_flatrag(q, doc_id, branches)
        if system == "raptor":
            return self._run_raptor(q, doc_id, branches)
        if system == "treerag_dfs":
            return self._run_treerag(q, doc_id, "dfs", branches)
        if system == "treerag_beam":
            return self._run_treerag(q, doc_id, "beam", branches)
        raise ValueError(f"Unknown system: {system}")

    # -- scoring ------------------------------------------------------------
    def score_answer(self, question, context, answer, expected) -> Dict[str, Any]:
        out = {
            "rouge_l": rouge_l_score(answer, expected),
            "bertscore": bertscore_f1(answer, expected, lang="ko"),
            "llm_judge": None,
        }
        if self.domain == "medical":
            from benchmarks.metrics.text_similarity import medical_entity_recall

            out["medical_entity_recall"] = medical_entity_recall(answer, expected)
        if self._judge is not None:
            out["llm_judge"] = self._judge.score_average(
                question, context, answer, expected
            )
        return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_dataset(dataset_arg: str) -> Dict[str, Any]:
    """Load a dataset by file path, or build a special benchmark by keyword.

    Keywords: ``hotpotqa`` (multi-hop) and ``medical`` (clinical) build their
    datasets on the fly; anything else is treated as a JSON file path.
    """
    key = str(dataset_arg).strip().lower()
    if key == "hotpotqa":
        from benchmarks.datasets.hotpotqa_loader import build_hotpotqa_dataset

        return build_hotpotqa_dataset()
    if key == "medical":
        path = _PROJECT_ROOT / "benchmarks" / "datasets" / "medical_benchmark.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(dataset_arg, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_mode(requested: str, gen_backend: str = "gemini") -> str:
    if requested in ("online", "offline"):
        return requested
    # When using the Ollama backend, skip the Gemini ping (override is already set).
    if gen_backend == "ollama":
        return "online"
    try:
        Config.CLIENT.models.generate_content(model=Config.MODEL_NAME, contents="ping")
        return "online"
    except Exception:
        return "offline"


def evaluate(dataset: Dict[str, Any], systems: List[str], evaluator: Evaluator,
             print_answers: bool = False,
             checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate all systems on the dataset.

    If ``checkpoint_path`` is given, completed systems are saved there after
    every system finishes. On restart with the same path, already-completed
    systems are loaded from the checkpoint and skipped, so only the remaining
    systems run.
    """
    questions = dataset["questions"]
    per_system: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}

    # Load checkpoint if available — resume from where we left off.
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and ckpt.is_file():
        try:
            saved = json.load(open(ckpt, encoding="utf-8")).get("per_question", {})
            for s in systems:
                if s in saved and len(saved[s]) == len(questions):
                    per_system[s] = saved[s]
                    print(f"  ✅ [checkpoint] {SYSTEM_LABELS.get(s, s)} — loaded {len(saved[s])} rows, skipping")
        except Exception as exc:
            print(f"  ⚠️  checkpoint load failed ({exc}), starting fresh")

    for system in systems:
        if per_system[system]:  # already loaded from checkpoint
            continue

        print(f"\n▶ Running system: {SYSTEM_LABELS.get(system, system)} "
              f"over {len(questions)} questions ...")
        for q in questions:
            expected = q.get("expected_answer_hint", "")
            t0 = time.perf_counter()
            try:
                answer, nodes = evaluator.run_system(
                    system, q["question"], q["document_id"]
                )
            except Exception as exc:
                answer, nodes = "", []
                print(f"   ⚠️  {system} failed on {q['question_id']}: {exc}")
            latency = time.perf_counter() - t0

            if print_answers:
                print(f"\n   ── [{system}] Q: {q['question'][:100]}")
                ans_preview = answer.replace("\n", " ")
                print(f"   ── A ({len(answer)} chars, {latency:.2f}s): {ans_preview[:300]}")
                if len(answer) < 10:
                    print(f"   ── ⛔ ANSWER TOO SHORT — smoke check FAIL")

            context = extractive_answer(nodes)
            scores = evaluator.score_answer(q["question"], context, answer, expected)
            per_system[system].append(
                {
                    "question_id": q["question_id"],
                    "document_id": q["document_id"],
                    "answer": answer,
                    "retrieved_count": len(nodes),
                    "context_tokens": int(len(context) / 4),
                    "latency": latency,
                    **scores,
                }
            )

        # Save checkpoint after each system completes.
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump({"per_question": per_system}, f, ensure_ascii=False)
            print(f"  💾 [checkpoint] saved after {SYSTEM_LABELS.get(system, system)} → {ckpt.name}")

    return per_system


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def aggregate(per_system: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    agg = {}
    for system, rows in per_system.items():
        judge_vals = [r["llm_judge"] for r in rows if r["llm_judge"] is not None]
        agg[system] = {
            "rouge_l": _mean([r["rouge_l"] for r in rows]),
            "bertscore": _mean([r["bertscore"] for r in rows]),
            "llm_judge": _mean(judge_vals) if judge_vals else None,
            "latency": _mean([r["latency"] for r in rows]),
            "context_tokens": _mean([r["context_tokens"] for r in rows]),
            "n": len(rows),
        }
        mer = [r["medical_entity_recall"] for r in rows if "medical_entity_recall" in r]
        if mer:
            agg[system]["medical_entity_recall"] = _mean(mer)
    return agg


def significance(per_system: Dict[str, List[Dict[str, Any]]], systems: List[str]) -> Dict[str, Any]:
    if "treerag_beam" not in systems:
        return {}
    st = StatisticalTests()
    beam = {r["question_id"]: r["rouge_l"] for r in per_system["treerag_beam"]}
    results = {}
    for system in systems:
        if system == "treerag_beam":
            continue
        base = {r["question_id"]: r["rouge_l"] for r in per_system[system]}
        common = [qid for qid in beam if qid in base]
        a = [beam[qid] for qid in common]
        b = [base[qid] for qid in common]
        res = st.paired_ttest(a, b)
        results[system] = {
            "p_value": res.p_value,
            "significant": res.significant,
            "effect_size": res.effect_size,
            "mean_diff": res.mean_difference,
        }
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(agg: Dict[str, Dict[str, float]], systems: List[str]) -> None:
    print("\n┌─────────────────┬─────────┬───────────┬───────────┬──────────┬────────┐")
    print("│ System          │ ROUGE-L │ BERTScore │ LLM-Judge │ Latency  │ CTX(K) │")
    print("├─────────────────┼─────────┼───────────┼───────────┼──────────┼────────┤")
    for system in systems:
        a = agg[system]
        lj = f"{a['llm_judge']:.2f}" if a["llm_judge"] else "  -  "
        print(
            f"│ {SYSTEM_LABELS.get(system, system):<15} │  {a['rouge_l']:.3f}  │"
            f"   {a['bertscore']:.3f}   │   {lj:^5} │  {a['latency']:.3f}s │"
            f" {a['context_tokens']/1000:>5.1f}  │"
        )
    print("└─────────────────┴─────────┴───────────┴───────────┴──────────┴────────┘")


def print_significance(sig: Dict[str, Any]) -> None:
    if not sig:
        return
    print("\nPaired t-test (TreeRAG-Beam vs baseline, ROUGE-L):")
    for system, r in sig.items():
        star = " *" if r["significant"] else ""
        print(
            f"  vs {SYSTEM_LABELS.get(system, system):<16} "
            f"p={r['p_value']:.4f}  Δ={r['mean_diff']:+.3f}  d={r['effect_size']:.2f}{star}"
        )


def save_markdown_table(
    agg: Dict[str, Dict[str, float]],
    sig: Dict[str, Any],
    systems: List[str],
    path: Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a GitHub-flavoured markdown results table to *path*."""
    lines: List[str] = []
    if meta:
        lines.append(f"# Benchmark Results — {meta.get('gen_backend','?')} / {meta.get('gen_model','?')}")
        lines.append(f"\n**Dataset**: {meta.get('dataset','')}  |  "
                     f"**Questions**: {meta.get('n_questions','')}  |  "
                     f"**Seed**: {meta.get('seed','')}  |  "
                     f"**Date**: {meta.get('date','')}\n")

    has_mer = any("medical_entity_recall" in agg.get(s, {}) for s in systems)
    has_judge = any(agg.get(s, {}).get("llm_judge") is not None for s in systems)

    header = "| System | ROUGE-L | BERTScore |"
    sep    = "|--------|---------|-----------|"
    if has_judge:
        header += " LLM-Judge |"
        sep    += "-----------|"
    if has_mer:
        header += " Med-Entity-Recall |"
        sep    += "-----------------|"
    header += " Latency(s) |"
    sep    += "-----------|"
    lines += [header, sep]

    for s in systems:
        a = agg.get(s, {})
        row = (f"| {SYSTEM_LABELS.get(s, s)} "
               f"| {a.get('rouge_l', 0):.3f} "
               f"| {a.get('bertscore', 0):.3f} |")
        if has_judge:
            lj = a.get("llm_judge")
            row += f" {lj:.3f} |" if lj is not None else " — |"
        if has_mer:
            mer = a.get("medical_entity_recall")
            row += f" {mer:.3f} |" if mer is not None else " — |"
        row += f" {a.get('latency', 0):.3f} |"
        lines.append(row)

    if sig:
        lines += [
            "",
            "## Significance (TreeRAG-Beam vs baselines, ROUGE-L paired t-test)",
            "",
            "| vs System | p-value | Δ mean | Cohen's d | Sig? |",
            "|-----------|---------|--------|-----------|------|",
        ]
        for system, r in sig.items():
            star = "✓" if r["significant"] else "✗"
            lines.append(
                f"| {SYSTEM_LABELS.get(system, system)} "
                f"| {r['p_value']:.4f} "
                f"| {r['mean_diff']:+.3f} "
                f"| {r['effect_size']:.2f} "
                f"| {star} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📄 Markdown → {path}")


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TreeRAG evaluation runner (PHASE C-3)")
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT / "benchmarks/datasets/full_benchmark.json"))
    parser.add_argument("--systems", default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--local-judge", action="store_true",
                        help="Use local Ollama model as judge (no API key needed)")
    parser.add_argument("--local-judge-model", default="llama3.1:8b",
                        help="Ollama model name for local judge (default: llama3.1:8b)")
    parser.add_argument("--mode", choices=["auto", "online", "offline"], default="auto")
    parser.add_argument("--domain", choices=["general", "medical"], default="general")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N questions (0 = all). "
                             "Useful for a cheap online direction-check.")
    parser.add_argument("--seed", type=int, default=0,
                        help="If set with --limit, randomly sample N questions with this seed.")
    parser.add_argument("--gen-backend", choices=["gemini", "ollama"], default="gemini",
                        help="LLM backend for answer generation (default: gemini). "
                             "Use 'ollama' for local generation via Ollama.")
    parser.add_argument("--gen-model", default="llama3.1:8b",
                        help="Model name for --gen-backend=ollama (default: llama3.1:8b).")
    args = parser.parse_args(argv)

    systems = ALL_SYSTEMS if args.systems == "all" else [
        s.strip() for s in args.systems.split(",") if s.strip()
    ]
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        parser.error(f"Unknown system(s): {unknown}. Choose from {ALL_SYSTEMS} or 'all'.")

    gen_backend: str = args.gen_backend
    gen_model: str = args.gen_model

    # ------------------------------------------------------------------ #
    # Ollama backend initialisation — must happen BEFORE detect_mode() so
    # the override is active for any test call detect_mode() might make.
    # ------------------------------------------------------------------ #
    if gen_backend == "ollama":
        from src.core.ollama_client import OllamaClient
        from src.config import set_client_override

        print(f"Gen backend     : Ollama  (model={gen_model})")
        try:
            ollama_client = OllamaClient(model=gen_model)
        except RuntimeError as exc:
            print(f"\n❌ {exc}")
            return 1
        set_client_override(ollama_client)
    else:
        print(f"Gen backend     : Gemini  (model={Config.MODEL_NAME})")

    dataset = load_dataset(args.dataset)

    if args.limit and args.limit > 0:
        qs = dataset["questions"]
        if args.seed:
            import random as _rnd

            qs = list(qs)
            _rnd.Random(args.seed).shuffle(qs)
        dataset["questions"] = qs[: args.limit]
        dataset["total_questions"] = len(dataset["questions"])

    mode = detect_mode(args.mode, gen_backend=gen_backend)
    gen_label = (f"Ollama {gen_model} reasoner"
                 if gen_backend == "ollama" else "real Gemini reasoner+judge")
    print(f"Evaluation mode : {mode.upper()}  "
          f"({'{}+'.format(gen_label) if mode == 'online' else 'offline keyword+extractive fallback'})")
    print(f"Systems         : {', '.join(systems)}")
    print(f"Questions       : {dataset.get('total_questions', len(dataset['questions']))}")

    evaluator = Evaluator(
        mode=mode,
        use_llm_judge=args.use_llm_judge,
        domain=args.domain,
        local_judge=args.local_judge,
        local_judge_model=args.local_judge_model,
        gen_backend=gen_backend,
        gen_model=gen_model,
    )
    # Print individual answers for small smoke runs (≤5 questions)
    n_questions = dataset.get("total_questions", len(dataset["questions"]))
    print_answers = bool(args.limit and args.limit <= 5)

    # Checkpoint path: <output_stem>.ckpt.json — survives kill/crash, auto-resumes.
    ckpt_path = None
    if args.output:
        ckpt_path = str(Path(args.output).with_suffix(".ckpt.json"))
    per_system = evaluate(dataset, systems, evaluator,
                          print_answers=print_answers, checkpoint_path=ckpt_path)
    agg = aggregate(per_system)
    sig = significance(per_system, systems)

    print_table(agg, systems)
    print_significance(sig)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else DEFAULT_REPORT_DIR / f"evaluation_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": ts,
        "mode": mode,
        "gen_backend": gen_backend,
        "gen_model": gen_model if gen_backend == "ollama" else Config.MODEL_NAME,
        "dataset": os.path.basename(args.dataset),
        "systems": systems,
        "summary": agg,
        "significance": sig,
        "per_question": per_system,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # Only update the stable copy when no explicit --output was given.
    if not args.output:
        latest = DEFAULT_REPORT_DIR / "evaluation_latest.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Report → {out_path}")

    # Save markdown table for paper use
    md_path = out_path.with_suffix(".md")
    save_markdown_table(
        agg, sig, systems, md_path,
        meta={
            "gen_backend": gen_backend,
            "gen_model": gen_model if gen_backend == "ollama" else Config.MODEL_NAME,
            "dataset": os.path.basename(args.dataset),
            "n_questions": dataset.get("total_questions", len(dataset["questions"])),
            "seed": args.seed,
            "date": ts,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
