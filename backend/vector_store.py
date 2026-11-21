"""
Thin wrapper around the Pinecone SDK.

The Pinecone SDK is called directly (instead of LangChain's PineconeVectorStore)
because the embed response reports `usage.total_tokens`. That number is what
lets us enforce the monthly token budget and measure characters-per-token.

Layout inside the index:
  namespace "<owner>/<repo>"   one per repository
      "<hash>-<i>"             chunk records   (metadata.kind = "chunk")
      "__repo_meta__"          completion marker, written LAST (kind = "meta")
  namespace "__system__"
      "usage-YYYY-MM"          embedding tokens spent on ingestion this month
"""
import logging
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from pinecone import Pinecone, ServerlessSpec

import config

log = logging.getLogger("repotrace.vector_store")

META_ID = "__repo_meta__"
SYSTEM_NS = "__system__"
# Marker records need a vector; cosine similarity needs a non-zero one.
_MARKER_VECTOR = [1.0] + [0.0] * (config.EMBED_DIM - 1)

_pc: Pinecone | None = None
_index = None
_init_lock = Lock()


def _client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc


def index():
    global _index
    with _init_lock:
        if _index is None:
            pc = _client()
            names = [i["name"] for i in pc.list_indexes()]
            if config.INDEX_NAME not in names:
                log.info("creating index %s", config.INDEX_NAME)
                pc.create_index(
                    name=config.INDEX_NAME,
                    dimension=config.EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
                )
            _index = pc.Index(config.INDEX_NAME)
    return _index


# ------------------------------------------------------------------ embeddings
def _get(obj, key, default=None):
    """Pinecone responses are objects in some SDK versions and dicts in others."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _TpmGate:
    """Keeps embedding calls under EMBED_TPM tokens in any rolling 60 s window."""

    def __init__(self):
        self.events: deque = deque()  # (timestamp, tokens)
        self.lock = Lock()

    def wait(self, tokens: int):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                if used + tokens <= config.EMBED_TPM or not self.events:
                    return
                sleep_for = 60 - (now - self.events[0][0]) + 0.1
            time.sleep(max(sleep_for, 0.1))

    def record(self, tokens: int):
        with self.lock:
            self.events.append((time.monotonic(), tokens))


_tpm = _TpmGate()


def embed(texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
    """
    Returns (vectors, tokens_used). input_type is "passage" for code chunks and
    "query" for questions (e5 is trained with different prefixes for each).

    On a 429 it retries after 2, 4, 8, 16, 32 s. Those add up to 62 s, more than
    the 60 s TPM window, so at least one full window passes before giving up.
    """
    estimate = int(sum(len(t) for t in texts) / config.CHARS_PER_TOKEN_EST)
    _tpm.wait(estimate)
    delay = 2
    for attempt in range(6):
        try:
            res = _client().inference.embed(
                model=config.EMBED_MODEL,
                inputs=texts,
                parameters={"input_type": input_type, "truncate": "END"},
            )
            break
        except Exception as e:  # SDK raises different classes per version
            if "429" not in str(e) or attempt == 5:
                raise
            log.warning("embed rate-limited, retrying in %ss", delay)
            time.sleep(delay)
            delay *= 2
    items = _get(res, "data")
    if items is None:
        items = list(res)
    vectors = [list(_get(item, "values")) for item in items]
    tokens = int(_get(_get(res, "usage"), "total_tokens", 0) or 0)
    _tpm.record(tokens or estimate)
    return vectors, tokens


# ------------------------------------------------------------------ writes
def delete_namespace(ns: str):
    try:
        index().delete(delete_all=True, namespace=ns)
    except Exception:
        pass  # namespace did not exist


def upsert_chunks(ns: str, records: list[dict]):
    for i in range(0, len(records), config.UPSERT_BATCH):
        index().upsert(vectors=records[i : i + config.UPSERT_BATCH], namespace=ns)


def write_meta(ns: str, meta: dict):
    index().upsert(
        vectors=[{"id": META_ID, "values": _MARKER_VECTOR, "metadata": {"kind": "meta", **meta}}],
        namespace=ns,
    )


def read_meta(ns: str) -> dict | None:
    res = index().fetch(ids=[META_ID], namespace=ns)
    vectors = _get(res, "vectors") or {}
    rec = vectors.get(META_ID)
    if rec is None:
        return None
    return dict(_get(rec, "metadata") or {})


# ------------------------------------------------------------------ monthly budget
def _month_id() -> str:
    return "usage-" + datetime.now(timezone.utc).strftime("%Y-%m")


def tokens_used_this_month() -> int:
    res = index().fetch(ids=[_month_id()], namespace=SYSTEM_NS)
    rec = (_get(res, "vectors") or {}).get(_month_id())
    if rec is None:
        return 0
    return int((_get(rec, "metadata") or {}).get("embed_tokens", 0))


_budget_lock = Lock()


def add_tokens_used(tokens: int):
    with _budget_lock:
        total = tokens_used_this_month() + tokens
        index().upsert(
            vectors=[{
                "id": _month_id(),
                "values": _MARKER_VECTOR,
                "metadata": {"kind": "usage", "embed_tokens": total},
            }],
            namespace=SYSTEM_NS,
        )


# ------------------------------------------------------------------ reads
def query(ns: str, vector: list[float], top_k: int) -> list[dict]:
    res = index().query(
        vector=vector,
        top_k=top_k,
        namespace=ns,
        include_metadata=True,
        filter={"kind": {"$eq": "chunk"}},
    )
    out = []
    for m in _get(res, "matches") or []:
        md = dict(_get(m, "metadata") or {})
        md["score"] = float(_get(m, "score", 0.0))
        out.append(md)
    return out
