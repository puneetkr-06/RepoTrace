"""
In-memory latency / ingestion metrics, exposed at GET /api/metrics.
These are the numbers to quote on a resume (p50 / p95, not a single run).
Every event is also logged as one JSON line so it survives in Render's logs.
"""
import json
import logging
import math
from collections import deque
from threading import Lock

import config

log = logging.getLogger("repotrace.metrics")

_lock = Lock()
_chat: deque = deque(maxlen=config.METRICS_WINDOW)
_ingest: deque = deque(maxlen=config.METRICS_WINDOW)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(s)))
    return s[rank - 1]


def record_chat(event: dict):
    with _lock:
        _chat.append(event)
    log.info(json.dumps({"event": "chat", **event}))


def record_ingest(event: dict):
    with _lock:
        _ingest.append(event)
    log.info(json.dumps({"event": "ingest", **event}))


def summary() -> dict:
    with _lock:
        chats = list(_chat)
        ingests = list(_ingest)

    def stats(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {
            "n": len(vals),
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
        }

    ok_ingests = [r for r in ingests if r.get("status") == "ready"]
    cpt = [r["chars_per_token"] for r in ok_ingests if r.get("chars_per_token")]
    return {
        "chat": {
            "retrieval_ms": stats("retrieval_ms", chats),
            "first_token_ms": stats("first_token_ms", chats),
            "total_ms": stats("total_ms", chats),
        },
        "ingest": {
            "count": len(ok_ingests),
            "duration_s": stats("duration_s", ok_ingests),
            "files": stats("files", ok_ingests),
            "embed_tokens": stats("embed_tokens", ok_ingests),
            # Measured characters per token across ingested repos.
            # Set CHARS_PER_TOKEN_EST to roughly the p5 of this.
            "chars_per_token": {
                "n": len(cpt),
                "p5": percentile(cpt, 5),
                "p50": percentile(cpt, 50),
            },
        },
    }
