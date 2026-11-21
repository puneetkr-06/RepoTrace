# RepoTrace

Chat with any public GitHub repository. RepoTrace indexes the code and answers
questions about it, citing the exact files and line ranges it used.

**Stack:** React + Vite + Tailwind (Vercel) · FastAPI (Render) · LangChain ·
Pinecone serverless + Pinecone Inference (`multilingual-e5-large`) · Gemini 2.5 Flash

## Architecture

```
React (Vercel)
  │  POST /api/ingest            ─► 200 ready  |  202 job
  │  GET  /api/repos/{repo}/status   (polled every 3 s)
  │  POST /api/chat              ─► streamed NDJSON
  ▼
FastAPI (Render)
  ├─ GitHub API      1 call for repo info, 1 for the commit SHA, 1 for the tarball
  ├─ Pinecone Inference   embeddings (passage / query)
  ├─ Pinecone index  one namespace per repo + a completion marker
  └─ Gemini          answer generation (LangChain LCEL, streamed)
```

### Ingestion (background job)

1. `get_repo_info` resolves the canonical name, rejects private repos, and reads
   the latest commit SHA of the default branch.
2. If the namespace has a `__repo_meta__` marker for that SHA, the repo is
   already indexed and the API answers `ready` immediately.
3. Otherwise a background job starts and the API returns `202`:
   - downloads the repo as **one tarball** and reads files in memory
     (skips vendored/build folders, binaries and files over `MAX_FILE_BYTES`)
   - splits each file with LangChain's language-aware splitter (20 languages)
     and records start/end line numbers
   - checks the token estimate against the per-repo cap and the monthly budget
   - embeds in batches of 96 (kept under 250K tokens/min) and upserts
   - writes the `__repo_meta__` marker **last**. A crash before that leaves
     no marker, so the next request re-ingests cleanly.

### Chat

1. Embeds the question (plus the previous question, for follow-ups).
2. Retrieves the top-k chunks from the repo's namespace, **once**.
3. Sends citations (permalinks to the ingested commit, with line ranges) first,
   then streams Gemini's answer token by token.

## Numbers and where they come from

| Setting | Value | Source / reasoning |
|---|---|---|
| Embedding dim / max tokens / batch | 1024 / 507 / 96 | Pinecone docs for `multilingual-e5-large` |
| Embedding tokens per month | 5,000,000 | Pinecone Starter plan |
| Embedding tokens per minute | 250,000 | Pinecone Starter plan |
| Query reserve | 500,000 | 10% of the month kept for question embeddings |
| Overlap cost | 1.25× | 1000 / (1000 − 200) |
| `MAX_REPO_TOKENS` | 250,000 | (5M − 0.5M) / 18 repos per month |
| `CHARS_PER_TOKEN_EST` | 2.5 → measure | Pessimistic start; set to measured p5 |
| `MAX_FILE_BYTES` | 100 KB → measure | 100K / 2.5 = 40K tokens = 16% of a repo budget; set to measured p99 |
| Chunk size check | 1000 chars | (1000 + 40 header) / 2.5 ≈ 416 tokens < 507 |
| Upsert batch | 96 | 96 × ~11 KB ≈ 1 MB < 2 MB request limit |
| Embed retry delays | 2,4,8,16,32 s | Sum 62 s > the 60 s TPM window |
| Chat per IP per day | 10 | Gemini RPD 250 / 25 expected users (set yours from AI Studio) |
| Global chat per minute | 10 | Gemini RPM |
| Ingest per IP per day | 2 | ~10% of the 18 repos/month |
| `TOP_K` | 6 → measure | Choose with `evaluate_retrieval.py` |
| Eval questions | ≥ 100 | 95% CI ±7.8 pts at p = 0.8 (±14.3 at n = 30) |
| `LLM_THINKING_BUDGET` | 0 | Measured: first token ≈ total time, so the wait was model "thinking" before writing |
| `FOLLOWUP_MAX_WORDS` | 8 → measure | Previous question is added to the search only for short questions with "it/this/that…" |
| Status poll | 3 s | ≤ 20 cheap in-memory reads/min while ingestion takes minutes |

### Measuring the rest

```bash
cd backend
# 1. chars/token, file-size percentiles, chunks over the 507-token limit
python scripts/measure_repos.py tiangolo/fastapi gin-gonic/gin expressjs/express
# 2. recall@k for k = 1..10 with 95% CI (repos must be ingested first)
cp scripts/eval_questions.example.jsonl scripts/eval_questions.jsonl   # then write ~100 questions
python scripts/evaluate_retrieval.py scripts/eval_questions.jsonl
# 3. p50 / p95 latency against the deployed API
python scripts/benchmark_latency.py --api https://repotrace-backend.onrender.com \
    --repo owner/name --questions scripts/eval_questions.jsonl --n 100
```

Live numbers from real traffic: `GET /api/metrics` (p50/p95 of retrieval,
first token and total time, plus measured chars/token per ingested repo).

## Local setup

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in the keys
uvicorn main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev                 # http://localhost:5173 (proxies /api to :8000)
```

Or run both at once from the repo root: `./run_local.sh`
(backend output goes to `backend/server.log`).
