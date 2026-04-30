# Promptfoo Step-1 PoC for TreeRAG

This directory contains a minimal Promptfoo integration that calls TreeRAG's `/chat` API through a custom Python provider.

## 1) Start TreeRAG API

Run backend first:

```bash
python main.py
```

Check health:

```bash
curl http://localhost:8000/health
```

## 2) Run Promptfoo eval

From repository root:

```bash
cd promptfoo
npx promptfoo@latest eval -c promptfooconfig.yaml
```

View results:

```bash
npx promptfoo@latest view
```

## 3) Optional environment variables

- `TREERAG_API_URL`: Chat endpoint URL (default: `http://localhost:8000/api/chat`)
- `TREERAG_TIMEOUT`: HTTP timeout seconds (default: `60`)

Example:

```bash
TREERAG_API_URL=http://localhost:8000/api/chat TREERAG_TIMEOUT=90 npx promptfoo@latest eval -c promptfooconfig.yaml
```

## 4) Notes

- `promptfooconfig.yaml` uses `file://./provider.py` as the provider.
- `tests/basic.yaml` is a starter test set. Add domain-specific cases as needed.
- `index_filenames` supports either:
  - comma-separated string
  - YAML list

Example list format in a test case:

```yaml
vars:
  question: "요약해줘"
  index_filenames:
    - "doc_a_index.json"
    - "doc_b_index.json"
```
