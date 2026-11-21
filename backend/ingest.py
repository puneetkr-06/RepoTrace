"""
Ingestion pipeline, run as a background job.

Flow:  GitHub tarball -> filter files -> language-aware chunks -> budget check
       -> embed (Pinecone Inference) -> upsert -> write completion marker.

The marker (__repo_meta__) is written only after every chunk is stored, so a
crash half-way leaves a namespace WITHOUT a marker, which the API treats as
"not ingested" and re-ingests from scratch. The old code had no marker, so a
half-finished namespace was reported as "Already ingested".
"""
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock

import chunking
import config
import keyword_search
import metrics
import vector_store
from github_source import RepoError, RepoInfo, load_files

log = logging.getLogger("repotrace.ingest")


@dataclass
class Job:
    repo: str
    commit_sha: str
    status: str = "queued"          # queued | running | ready | failed
    stage: str = "queued"           # downloading | chunking | embedding | finalizing
    files: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    est_tokens: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("started_at")
        return d


_jobs: dict[str, Job] = {}
_jobs_lock = Lock()


def get_job(repo: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(repo)


def try_start(info: RepoInfo) -> tuple[Job, bool]:
    """Registers a job unless one is already running. Returns (job, created)."""
    with _jobs_lock:
        job = _jobs.get(info.full_name)
        if job and job.status in ("queued", "running"):
            return job, False
        job = Job(repo=info.full_name, commit_sha=info.commit_sha)
        _jobs[info.full_name] = job
        return job, True


def run(info: RepoInfo, job: Job):
    t0 = time.time()
    ns = info.full_name
    event = {"repo": ns, "commit": info.commit_sha}
    try:
        job.status = "running"

        # Raw text above this can never fit the token cap, even before overlap
        # and headers are added: chars <= MAX_REPO_TOKENS * chars_per_token / overlap.
        max_chars = int(config.MAX_REPO_TOKENS * config.CHARS_PER_TOKEN_EST / config.OVERLAP_FACTOR)

        job.stage = "downloading"
        files, stats = load_files(info, max_chars=max_chars)
        if not files:
            raise RepoError("No supported source files found in this repository.")
        job.files = stats.files_kept

        job.stage = "chunking"
        chunks = chunking.chunk_files(files)
        job.chunks_total = len(chunks)
        job.est_tokens = chunking.estimate_tokens(chunks)

        if job.est_tokens > config.MAX_REPO_TOKENS:
            raise RepoError(
                f"Repository needs ~{job.est_tokens:,} embedding tokens; "
                f"the per-repo limit is {config.MAX_REPO_TOKENS:,}.",
                413,
            )
        used = vector_store.tokens_used_this_month()
        available = config.MONTHLY_EMBED_TOKENS - config.QUERY_TOKEN_RESERVE - used
        if job.est_tokens > available:
            raise RepoError(
                "Monthly free embedding budget is used up "
                f"({used:,} tokens spent). Try again next month.",
                429,
            )

        job.stage = "embedding"
        vector_store.delete_namespace(ns)  # also removes any old marker
        total_tokens = 0
        for i in range(0, len(chunks), config.EMBED_BATCH):
            batch = chunks[i : i + config.EMBED_BATCH]
            vectors, tokens = vector_store.embed([c.embed_text for c in batch], "passage")
            total_tokens += tokens
            vector_store.upsert_chunks(ns, [
                {
                    "id": c.id,
                    "values": v,
                    "metadata": {
                        "kind": "chunk",
                        "path": c.path,
                        "language": c.language,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "code": c.code,
                    },
                }
                for c, v in zip(batch, vectors)
            ])
            job.chunks_done += len(batch)

        job.stage = "finalizing"
        vector_store.add_tokens_used(total_tokens)
        embedded_chars = sum(len(c.embed_text) for c in chunks)
        chars_per_token = round(embedded_chars / total_tokens, 3) if total_tokens else None
        meta = {
            "status": "ready",
            "chunker": chunking.active_chunker(),
            "commit_sha": info.commit_sha,
            "branch": info.default_branch,
            "files": stats.files_kept,
            "chunks": len(chunks),
            "embed_tokens": total_tokens,
            "embedded_chars": embedded_chars,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        vector_store.write_meta(ns, meta)  # LAST step = the repo is complete
        # Warm the keyword index from the chunks still in memory (no Pinecone reads).
        keyword_search.prime(ns, f"{meta['commit_sha']}:{meta['chunker']}", [
            {"id": c.id, "path": c.path, "language": c.language, "start_line": c.start_line,
             "end_line": c.end_line, "code": c.code}
            for c in chunks
        ])
        job.status = "ready"
        job.stage = "done"
        event.update(
            status="ready",
            chunker=chunking.active_chunker(),
            files=stats.files_kept,
            skipped_size=stats.files_skipped_size,
            skipped_binary=stats.files_skipped_binary,
            chunks=len(chunks),
            est_tokens=job.est_tokens,
            embed_tokens=total_tokens,
            chars_per_token=chars_per_token,
            tarball_bytes=stats.tarball_bytes,
        )
    except RepoError as e:
        job.status, job.error = "failed", str(e)
        event.update(status="failed", error=str(e))
    except Exception as e:  # unexpected: log the trace, show a short message
        log.exception("ingestion failed for %s", ns)
        job.status, job.error = "failed", f"Ingestion failed: {e}"
        event.update(status="failed", error=str(e))
    finally:
        event["duration_s"] = round(time.time() - t0, 2)
        metrics.record_ingest(event)
