import json
import logging
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import chunking
import config
import ingest
import metrics
import keyword_search
import rag
import rate_limit
import vector_store
from github_source import RepoError, get_repo_info, load_files, parse_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("repotrace.api")

app = FastAPI(title="RepoTrace API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["*"],
)


@app.exception_handler(RepoError)
def repo_error_handler(_: Request, exc: RepoError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# Small cache so each chat does not re-fetch the marker from Pinecone.
_ready_meta: dict[str, dict] = {}


def _ready(repo: str) -> dict | None:
    meta = _ready_meta.get(repo)
    if meta is None:
        meta = vector_store.read_meta(repo)
        if meta and meta.get("status") == "ready":
            _ready_meta[repo] = meta
    return meta


# ---------------------------------------------------------------- endpoints
# Endpoints are plain `def`, not `async def`: they call blocking SDKs, and
# FastAPI runs `def` endpoints in a thread pool so the event loop stays free.

@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


class IngestRequest(BaseModel):
    repo_url: str = Field(max_length=300)


@app.post("/api/ingest")
def start_ingest(body: IngestRequest, request: Request, background: BackgroundTasks):
    """
    Returns at once. 200 + status "ready" if this exact commit is already
    indexed, otherwise 202 + a job the client polls at /api/repos/{repo}/status.
    """
    info = get_repo_info(parse_repo(body.repo_url))
    repo = info.full_name

    meta = vector_store.read_meta(repo)
    if (meta and meta.get("status") == "ready" and meta.get("commit_sha") == info.commit_sha
            and meta.get("chunker") == chunking.active_chunker()):
        _ready_meta[repo] = meta
        return {"status": "ready", "repo": repo, "commit_sha": info.commit_sha}

    running = ingest.get_job(repo)
    if running and running.status in ("queued", "running"):
        return JSONResponse(status_code=202, content=running.public())

    # Only real work counts against the per-IP limit; cached repos are free.
    if not rate_limit.is_admin(request):
        rate_limit.enforce(rate_limit.ingest_per_ip, rate_limit.client_ip(request), "repository ingestion")

    job, created = ingest.try_start(info)
    if created:
        _ready_meta.pop(repo, None)  # new commit: old cache entry is stale
        background.add_task(ingest.run, info, job)
    return JSONResponse(status_code=202, content=job.public())


@app.get("/api/repos/{owner}/{name}/status")
def repo_status(owner: str, name: str):
    repo = parse_repo(f"{owner}/{name}")
    job = ingest.get_job(repo)
    if job and job.status != "ready":
        return job.public()
    meta = None
    try:
        meta = _ready(repo)
    except Exception:
        log.exception("status lookup failed")
    if meta and meta.get("status") == "ready":
        return {"status": "ready", "repo": repo, **meta}
    # Name casing may differ from GitHub's canonical full_name.
    raise HTTPException(status_code=404, detail="Repository not ingested yet.")


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    repo: str
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    """
    Streams newline-delimited JSON:
      {"type":"citations","citations":[...]}   first
      {"type":"token","text":"..."}            many
      {"type":"done"} or {"type":"error","detail":"..."}
    """
    repo = parse_repo(body.repo)
    meta = _ready(repo)
    if not meta or meta.get("status") != "ready":
        raise HTTPException(status_code=404, detail="Repository not ingested yet.")

    if not rate_limit.is_admin(request):
        rate_limit.enforce(rate_limit.chat_per_ip, rate_limit.client_ip(request), "chat messages today")
    rate_limit.enforce(rate_limit.chat_global_minute, "global", "the AI model (per minute)")
    rate_limit.enforce(rate_limit.chat_global_day, "global", "the AI model (today)")

    history = [m.model_dump() for m in body.history]
    t0 = time.perf_counter()
    chunks, query_tokens = rag.retrieve(repo, body.message, history, version=rag.index_version(meta))
    retrieval_ms = (time.perf_counter() - t0) * 1000

    def generate():
        first_token_ms = None
        status = "ok"
        try:
            cites = rag.citations(repo, meta["commit_sha"], chunks)
            yield json.dumps({"type": "citations", "citations": cites}) + "\n"
            for piece in rag.stream_answer(repo, body.message, history, chunks):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                yield json.dumps({"type": "token", "text": piece}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            status = "error"
            log.exception("chat failed")
            msg = "The AI model is rate-limited. Try again in a minute." if "429" in str(e) else "Failed to generate an answer."
            yield json.dumps({"type": "error", "detail": msg}) + "\n"
        finally:
            metrics.record_chat({
                "repo": repo,
                "status": status,
                "retrieval_ms": round(retrieval_ms, 1),
                "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "query_tokens": query_tokens,
                "chunks": len(chunks),
            })

    return StreamingResponse(generate(), media_type="application/x-ndjson")


class RetrieveRequest(BaseModel):
    repo: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str | None = None  # "dense" | "bm25" | "hybrid"; default config.RETRIEVAL_MODE


def _eval_rows(chunks: list[dict]) -> list[dict]:
    return [
        {"id": c.get("id"), "path": c["path"], "start_line": int(c["start_line"]),
         "end_line": int(c["end_line"]), "score": round(c["score"], 4)}
        for c in chunks
    ]


@app.post("/api/eval/retrieve")
def eval_retrieve(body: RetrieveRequest, request: Request):
    """
    Retrieval only (no Gemini call) for the evaluation scripts.
    Local requests only; returns 404 anywhere else.
    """
    if not rate_limit.is_local(request):
        raise HTTPException(status_code=404, detail="Not found")
    repo = parse_repo(body.repo)
    meta = _ready(repo) or {}
    t0 = time.perf_counter()
    chunks, tokens = rag.retrieve(repo, body.question, [], top_k=body.top_k,
                                  version=rag.index_version(meta), mode=body.mode)
    return {
        "mode": body.mode or config.RETRIEVAL_MODE,
        "query_tokens": tokens,
        "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1),
        "results": _eval_rows(chunks),
    }


@app.post("/api/eval/candidates")
def eval_candidates(body: RetrieveRequest, request: Request):
    """
    The raw vector list and the raw BM25 list (top_k each) for one question, so
    fusion settings can be tuned offline without re-embedding. Local only.
    """
    if not rate_limit.is_local(request):
        raise HTTPException(status_code=404, detail="Not found")
    repo = parse_repo(body.repo)
    meta = _ready(repo) or {}
    dense, tokens = rag.retrieve(repo, body.question, [], top_k=body.top_k, mode="dense")
    bm = keyword_search.index_for(repo, rag.index_version(meta))
    return {
        "query_tokens": tokens,
        "bm25_build_ms": getattr(bm, "build_ms", None),
        "bm25_chunks": len(bm.chunks),
        "dense": _eval_rows(dense),
        "bm25": _eval_rows(bm.search(body.question, body.top_k)),
    }


@app.get("/api/eval/chunk-stats")
def eval_chunk_stats(repo: str, request: Request):
    """
    Chunks a repo with BOTH chunkers (no embedding, no tokens spent) and reports
    chunk counts, estimated tokens and how many definitions stay whole.
    Local requests only.
    """
    if not rate_limit.is_local(request):
        raise HTTPException(status_code=404, detail="Not found")
    info = get_repo_info(parse_repo(repo))
    files, _ = load_files(info)
    out = {"repo": info.full_name, "ast_available": chunking.AST_AVAILABLE,
           "ast_extensions": chunking.ast_languages(), "files": len(files)}
    for name in ("splitter", "ast"):
        chunks = chunking.chunk_files(files, chunker=name)
        sizes = sorted(len(c.code) for c in chunks)
        out[name] = {
            "chunks": len(chunks),
            "est_tokens": chunking.estimate_tokens(chunks),
            "median_chunk_chars": sizes[len(sizes) // 2] if sizes else 0,
            **chunking.integrity(files, chunks),
        }
    return out


@app.get("/api/eval/file-chunks")
def eval_file_chunks(repo: str, path: str, request: Request):
    """Line ranges each chunker produces for one file (debugging). Local only."""
    if not rate_limit.is_local(request):
        raise HTTPException(status_code=404, detail="Not found")
    info = get_repo_info(parse_repo(repo))
    files, _ = load_files(info)
    f = next((x for x in files if x.path == path), None)
    if f is None:
        raise HTTPException(status_code=404, detail="File not in the ingested set")
    return {
        "lines": f.text.count("\n") + 1,
        **{name: [[c.start_line, c.end_line, len(c.code)] for c in chunking.chunk_file(f, name)]
           for name in ("splitter", "ast")},
    }


@app.get("/api/eval/index-stats")
def eval_index_stats(request: Request):
    """Vector count per namespace, to confirm an index holds only fresh chunks. Local only."""
    if not rate_limit.is_local(request):
        raise HTTPException(status_code=404, detail="Not found")
    stats = vector_store.index().describe_index_stats()
    namespaces = vector_store._get(stats, "namespaces") or {}
    return {
        "total": vector_store._get(stats, "total_vector_count"),
        "namespaces": {
            name: vector_store._get(ns, "vector_count") for name, ns in namespaces.items()
        },
        "ast_extensions": chunking.ast_languages(),
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics.summary()
