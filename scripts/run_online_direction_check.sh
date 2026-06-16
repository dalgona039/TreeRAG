#!/usr/bin/env bash
#
# One-shot ONLINE direction check (Step 1 of the reviewer feedback).
#
# Runs 5 questions through BM25, FlatRAG and TreeRAG-Beam with the REAL Gemini
# backend + LLM-judge, so you can see whether TreeRAG actually beats the flat
# baselines on generation quality (LLM-Judge) and citation traceability.
#
# Requirements on the machine that runs this:
#   - Open internet (Gemini API reachable)
#   - A valid GOOGLE_API_KEY in ./.env   (already present in this repo)
#   - python3 (3.10+; 3.12 recommended)
#
# Usage:
#   bash scripts/run_online_direction_check.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Creating an isolated virtualenv (.venv_verify)"
python3 -m venv .venv_verify
# shellcheck disable=SC1091
source .venv_verify/bin/activate

echo "==> Installing dependencies (this may take a minute)"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q rouge-score rank-bm25 scipy

echo "==> Sanity check: can we reach Gemini with the project Config?"
python3 - <<'PY'
import sys
from dotenv import load_dotenv
load_dotenv(".env")
try:
    from src.config import Config
    r = Config.CLIENT.models.generate_content(model=Config.MODEL_NAME,
                                               contents="Reply with the single word OK")
    print("   Gemini OK via model:", Config.MODEL_NAME, "->", (r.text or "").strip()[:30])
except Exception as e:
    print("   GEMINI CALL FAILED:", repr(e)[:200])
    print("   If this is a model-name error, edit src/config.py MODEL_NAME to an")
    print("   accessible model (e.g. 'gemini-2.5-flash') and re-run.")
    sys.exit(1)
PY

echo "==> Running 5-question ONLINE direction check"
python3 benchmarks/run_real_evaluation.py \
  --systems bm25,flatrag,treerag_beam \
  --limit 5 --mode online --use-llm-judge \
  --output data/benchmark_reports/direction_check.json

echo
echo "==> Done. The table above is the direction check."
echo "    Look at LLM-Judge (not just ROUGE-L) and whether TreeRAG-Beam answers"
echo "    carry [doc, p.X] citations. Paste the table back to continue."
