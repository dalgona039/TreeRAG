"""B5 — recompute ROUGE-L and BERTScore-proxy as separate P/R/F variants.

Reads an already-stored fair-protocol result file (per_question hypotheses) plus the
benchmark dataset's gold `expected_answer_hint` (joined by question_id), and recomputes
metrics using precision, recall, and F separately — instead of only the F-measure the
pipeline currently records. Does not re-run any LLM calls and does not overwrite any
existing result file; only reads existing JSON and writes a new CSV.

Usage:
    python -m benchmarks.analysis.b5_recompute_metric_variants \
        --result data/benchmark_reports/online_local_llama_general_v4_n100.json \
        --dataset benchmarks/datasets/full_benchmark.json \
        --out docs/paper_revision/B5_metric_variants_n100.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path

from benchmarks.metrics.text_similarity import _rouge_scorer, _tokens  # reuse the exact tokenizer/scorer used in the pipeline


def token_f1_prf(hypothesis: str, reference: str):
    """Same multiset overlap as text_similarity._token_f1, but returns P, R, F separately."""
    hyp, ref = _tokens(hypothesis), _tokens(reference)
    if not hyp and not ref:
        return 1.0, 1.0, 1.0
    if not hyp or not ref:
        return 0.0, 0.0, 0.0
    common = Counter(hyp) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    prec, rec = overlap / len(hyp), overlap / len(ref)
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def rouge_l_prf(hypothesis: str, reference: str):
    if not hypothesis and not reference:
        return 1.0, 1.0, 1.0
    if not hypothesis or not reference:
        return 0.0, 0.0, 0.0
    result = _rouge_scorer().score(reference, hypothesis)
    r = result["rougeL"]
    return float(r.precision), float(r.recall), float(r.fmeasure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    gold_by_qid = {q["question_id"]: q.get("expected_answer_hint", "") for q in dataset["questions"]}

    rows = []
    per_system_lengths = {}
    per_system_variants = {}

    for system, records in result["per_question"].items():
        lengths = []
        rouge_p, rouge_r, rouge_f = [], [], []
        bert_p, bert_r, bert_f = [], [], []
        for rec in records:
            qid = rec["question_id"]
            gold = gold_by_qid.get(qid, "")
            hyp = rec.get("answer", "") or ""
            if not gold:
                continue
            lengths.append(len(hyp.split()))
            rp, rr, rf = rouge_l_prf(hyp, gold)
            bp, br, bf = token_f1_prf(hyp, gold)
            rouge_p.append(rp); rouge_r.append(rr); rouge_f.append(rf)
            bert_p.append(bp); bert_r.append(br); bert_f.append(bf)
            rows.append({
                "system": system, "question_id": qid,
                "answer_len_words": len(hyp.split()),
                "rouge_l_precision": rp, "rouge_l_recall": rr, "rouge_l_f": rf,
                "bertscore_proxy_precision": bp, "bertscore_proxy_recall": br, "bertscore_proxy_f": bf,
                "pipeline_reported_rouge_l": rec.get("rouge_l"),
                "pipeline_reported_bertscore": rec.get("bertscore"),
            })
        per_system_lengths[system] = lengths
        per_system_variants[system] = {
            "rouge_l_precision": stats.mean(rouge_p) if rouge_p else None,
            "rouge_l_recall": stats.mean(rouge_r) if rouge_r else None,
            "rouge_l_f": stats.mean(rouge_f) if rouge_f else None,
            "bertscore_proxy_precision": stats.mean(bert_p) if bert_p else None,
            "bertscore_proxy_recall": stats.mean(bert_r) if bert_r else None,
            "bertscore_proxy_f": stats.mean(bert_f) if bert_f else None,
            "n": len(rouge_p),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print()
    print("=== Per-system summary (means) ===")
    for system, v in per_system_variants.items():
        lens = per_system_lengths[system]
        mean_len = stats.mean(lens) if lens else float("nan")
        std_len = stats.pstdev(lens) if len(lens) > 1 else 0.0
        print(f"\n{system} (n={v['n']}, answer len mean={mean_len:.1f} words, std={std_len:.1f}):")
        print(f"  ROUGE-L      P={v['rouge_l_precision']:.4f}  R={v['rouge_l_recall']:.4f}  F={v['rouge_l_f']:.4f}")
        print(f"  BERTScore*   P={v['bertscore_proxy_precision']:.4f}  R={v['bertscore_proxy_recall']:.4f}  F={v['bertscore_proxy_f']:.4f}   (*token-F1 proxy — see B5 doc)")


if __name__ == "__main__":
    main()
