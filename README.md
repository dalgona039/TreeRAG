
# TreeRAG

TreeRAG is a hierarchical RAG system that indexes PDF documents into tree-structured JSON and answers questions with page-traceable citations.

The project includes:
- FastAPI backend for upload, indexing, chat, tree, cache, and graph APIs
- TreeRAG reasoner with DFS/Beam traversal, optional contextual compression, and optional cross-reference resolving
- Next.js frontend for multi-document chat and tree exploration
- Benchmark/evaluation scripts for BM25, FlatRAG, TreeRAG-DFS, TreeRAG-Beam, and related baselines

## Why TreeRAG

Traditional flat retrieval often loses section hierarchy. TreeRAG keeps document structure and traverses only relevant branches.

Core ideas:
- Build a page-aware hierarchical index from PDFs
- Route query to one or more indexed documents
- Traverse tree nodes (DFS or Beam Search)
- Generate answers with source grounding and citation extraction
- Cache responses and enforce API rate limits for stable serving

## Key Features

- Multi-document querying with automatic document routing
- Deep traversal mode (TreeNavigator or BeamSearchNavigator)
- Optional contextual compression before final generation
- Citation extraction from generated answers
- Hallucination warning layer based on node-grounding confidence
- In-memory response cache with stats and clear endpoints
- API rate limiting (SlowAPI)
- Optional async indexing task endpoints with Celery/Redis
- Reasoning graph build/load endpoints

## Tech Stack

Backend:
- Python
- FastAPI
- google-genai (Gemini API)
- pypdf
- slowapi
- Redis + Celery (optional async task queue)

Frontend:
- Next.js 16
- React 19
- TypeScript

Evaluation:
- pytest
- benchmark scripts under benchmarks/

## Repository Layout

- src/
  - api/: REST routes and request/response models
  - core/: indexer, reasoner, traversal, beam search, compressors, baselines
  - middleware/: security headers
  - repositories/: session persistence
  - utils/: cache, file validation, hallucination detection
- benchmarks/: evaluation runners, datasets, analysis, metrics
- frontend/: Next.js application
- data/
  - raw/: uploaded PDFs
  - indices/: generated index JSON files and graph files
  - benchmark_reports/: benchmark outputs

## Prerequisites

- Python 3.11+ (tested in this repository with Python 3.13)
- Node.js 20+ for frontend
- Gemini API key

## Quick Start (Local)

### 1) Create environment and install backend dependencies

```bash
cd /Volumes/a3122a1/TreeRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For benchmark scripts that need extra metrics packages:

```bash
pip install rouge-score rank-bm25 scipy
```

### 2) Configure environment variables

Create .env in repository root:

```bash
GOOGLE_API_KEY=your_key_here
```

Optional Gemini resiliency controls (recommended on free tier):

```bash
GEMINI_MIN_INTERVAL_S=13
GEMINI_MAX_RETRIES=5
```

Notes:
- GEMINI_MIN_INTERVAL_S throttles request spacing to reduce 429 bursts
- The code also retries quota-style errors with backoff in src/config.py

### 3) Run backend

```bash
source .venv/bin/activate
python main.py
```

Backend endpoints:
- API base: http://localhost:8000/api
- OpenAPI docs: http://localhost:8000/docs
- Health: http://localhost:8000/health and /health/deep

### 4) Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: http://localhost:3000

## Quick Start (Docker)

```bash
cd /Volumes/a3122a1/TreeRAG
# ensure .env has GOOGLE_API_KEY
docker-compose up --build -d
```

Services:
- frontend: 3000
- backend: 8000
- redis: 6379
- celery-worker (optional worker service included in compose)

See DOCKER.md for operational commands.

## Basic API Flow

1. Upload PDF

```bash
curl -F "file=@/absolute/path/to/doc.pdf" http://localhost:8000/api/upload
```

2. Create index

```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"filename":"<uploaded_filename>.pdf"}'
```

3. Query chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question":"핵심 내용을 요약해줘",
    "index_filenames":["<uploaded_filename>_index.json"],
    "use_deep_traversal":true,
    "max_depth":5,
    "max_branches":3,
    "domain_template":"general",
    "language":"ko"
  }'
```

## Main API Endpoints

Document and chat:
- GET /api/
- POST /api/upload
- POST /api/index
- POST /api/chat
- GET /api/indices
- GET /api/pdfs
- GET /api/tree/{index_filename}
- GET /api/pdf/{filename}

Sessions:
- GET /api/sessions
- PUT /api/sessions

Cache:
- GET /api/cache/stats
- POST /api/cache/clear

Reasoning graph:
- POST /api/graph/build/{document_name}
- GET /api/graph/{document_name}

Async tasks (requires Celery availability):
- POST /api/tasks/index
- POST /api/tasks/index/batch
- GET /api/tasks/{task_id}
- DELETE /api/tasks/{task_id}
- GET /api/tasks/

## Traversal and Reasoning Behavior

Reasoner class: src/core/reasoner.py

- Traversal algorithms:
  - DFS (TreeNavigator)
  - Beam Search (BeamSearchNavigator)
- Optional contextual compression
- Optional reference resolver for section/chapter references
- Domain prompt templates: general, medical, legal, financial, academic
- Language instructions: ko/en/ja
- Cache key includes prompt cache version and traversal settings

## Benchmark and Evaluation

Primary runner:
- benchmarks/run_real_evaluation.py

Example (cheap direction check):

```bash
source .venv/bin/activate
export GEMINI_MIN_INTERVAL_S=13
python benchmarks/run_real_evaluation.py \
  --systems bm25,flatrag,treerag_beam \
  --limit 3 --mode online \
  --output data/benchmark_reports/direction_check.json
```

Script helper:

```bash
bash scripts/run_online_direction_check.sh
```

Caution:
- Online mode consumes Gemini quota quickly, especially TreeRAG-Beam
- Free-tier limits can produce 429/503 and partially empty answers
- If result rows are empty due to quota failures, treat that run as invalid for model quality comparison

## Testing

Run all tests:

```bash
source .venv/bin/activate
pytest
```

Run specific tests:

```bash
pytest tests/test_config_resilient.py -v
```

pytest config is in pytest.ini.

## Security and Validation

Implemented safeguards include:
- Upload file validation (extension, MIME, size, path traversal)
- Security headers middleware
- API rate limiting
- Session persistence via repository layer

For local development hygiene, run:

```bash
bash setup-git-hooks.sh
```

## Troubleshooting

### 1) 429 RESOURCE_EXHAUSTED

Symptoms:
- Empty or failed responses in benchmarks
- frequent quota errors in logs

Actions:
- set GEMINI_MIN_INTERVAL_S=13 (or larger)
- reduce benchmark limit and/or systems
- disable LLM judge for cheap checks
- retry after quota reset if daily limit exceeded

### 2) 503 UNAVAILABLE

Symptoms:
- transient model overload errors

Actions:
- rerun later (service-side congestion)
- keep retries enabled (GEMINI_MAX_RETRIES)

### 3) No indexed documents found

Actions:
- verify upload succeeded to data/raw
- call /api/index for each uploaded PDF
- check files under data/indices

### 4) Frontend cannot call backend

Actions:
- ensure backend is running on port 8000
- verify NEXT_PUBLIC_API_BASE_URL when using Docker/production

## License

MIT (see LICENSE).

## Acknowledgements

This project is influenced by tree-structured and vectorless RAG approaches, and is developed as an engineering/research codebase for hierarchical document reasoning and evaluation.
