import json
import logging
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
import ingest
import metrics
import rag
import rate_limit
import vector_store
from github_source import RepoError, get_repo_info, parse_repo

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
    if meta and meta.get("status") == "ready" and meta.get("commit_sha") == info.commit_sha:
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
    chunks, query_tokens = rag.retrieve(repo, body.message, history)
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


@app.get("/api/metrics")
def get_metrics():
    return metrics.summary()
