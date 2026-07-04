"""
Re-judge already-generated answers with qwen3.5:9b (think=False fix applied).
Reads answers from online_local_qwen_judge_n100.json,
looks up question/expected from full_benchmark.json,
calls LocalJudge, and writes updated llm_judge scores in-place.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.metrics.llm_judge import LocalJudge

RESULT_FILE = ROOT / "data/benchmark_reports/online_local_qwen_judge_n100.json"
BENCH_FILE  = ROOT / "benchmarks/datasets/full_benchmark.json"

def main():
    result = json.load(open(RESULT_FILE, encoding="utf-8"))
    bench  = json.load(open(BENCH_FILE,  encoding="utf-8"))

    # question_id → {question, expected_answer_hint}
    questions = bench if isinstance(bench, list) else bench.get("questions", [])
    qmap = {q["question_id"]: q for q in questions}

    judge = LocalJudge(model="qwen3.5:9b")

    systems = list(result["per_question"].keys())
    total = sum(len(rows) for rows in result["per_question"].values())
    done = 0

    for sys_name in systems:
        rows = result["per_question"][sys_name]
        updated = 0
        for row in rows:
            qid = row.get("question_id")
            bq  = qmap.get(qid, {})
            question = bq.get("question", "")
            expected = bq.get("expected_answer_hint", "")
            answer   = row.get("answer", "")

            score = judge.score_average(
                question=question,
                context="",   # context not stored; omitted for re-judging
                answer=answer,
                expected=expected,
            )
            row["llm_judge"] = score
            if score is not None:
                updated += 1
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} done  (last score: {score})", flush=True)

        print(f"[{sys_name}] {updated}/{len(rows)} scored", flush=True)

    json.dump(result, open(RESULT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved → {RESULT_FILE}")

if __name__ == "__main__":
    main()
