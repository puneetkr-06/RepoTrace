"""
Keyword search (BM25) over one repo's chunks, and Reciprocal Rank Fusion (RRF)
to merge it with vector search.

Why: vector search matches meaning, so it misses questions that hinge on an
exact name ("where is ConsistentHashRing used?", "what does MAX_RETRIES do?")
and lets similar-sounding files (a frontend page about "bookings") outrank the
backend file that actually contains the logic. BM25 scores exact word overlap,
weighted by how rare each word is in this repo, which is the opposite failure
profile. RRF combines the two rankings without having to compare their scores.

Where the index comes from: the chunk text is already stored in Pinecone
metadata at ingestion, so the BM25 index is built in memory from Pinecone the
first time a repo is searched (one list + a few fetch calls) and cached. No
extra database, no extra embedding tokens.
"""
import math
import re
import threading
import time
from collections import Counter, OrderedDict

import logging

import config
import vector_store

log = logging.getLogger("repotrace.keyword")

# Common English words in questions ("how does the ... work") carry no signal.
_STOP = set("""a an and are as at be by can do does for from has have how i if in
into is it its of on or so that the their them then there these this those to
was what when where which who why will with would you your""".split())

_WORD = re.compile(r"[A-Za-z0-9_]+")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """
    Code-aware tokens. "getUserById" -> getuserbyid, get, user, by, id;
    "MAX_FILE_BYTES" -> max_file_bytes, max, file, bytes. The whole identifier
    lets an exact name match strongly; the parts let "user id" match it too.
    """
    out = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if len(lw) > 1 and lw not in _STOP:
            out.append(lw)
        parts = [p.lower() for p in _PARTS.findall(w)]
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) > 1 and p not in _STOP)
    return out


def _doc_text(c: dict) -> str:
    # The path is part of the document: "LoadBalancer.cpp" is strong evidence
    # for a question about the load balancer.
    return f"{c['path']}\n{c['code']}"


class BM25:
    """Okapi BM25 (Robertson & Zaragoza 2009), Lucene's non-negative IDF."""

    def __init__(self, chunks: list[dict], k1: float, b: float):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(_doc_text(c))) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        n = len(chunks)
        self.avgdl = (sum(self.len) / n) if n else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, top_k: int) -> list[dict]:
        q = [w for w in set(tokenize(query)) if w in self.idf]
        if not q:
            return []
        scored = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": s} for s, i in scored[:top_k]]


# ns -> (version, BM25). LRU so a long-running server does not keep every repo
# ever searched; 18 repos/month is the ingestion budget, so 32 covers them all.
_cache: "OrderedDict[str, tuple[str, BM25]]" = OrderedDict()
_lock = threading.Lock()
_CACHE_SIZE = 32


def index_for(ns: str, version: str) -> BM25:
    """version = commit SHA + chunker; a re-ingested repo gets a fresh index."""
    with _lock:
        hit = _cache.get(ns)
        if hit and hit[0] == version:
            _cache.move_to_end(ns)
            return hit[1]
    t0 = time.perf_counter()
    chunks = vector_store.all_chunks(ns)
    t1 = time.perf_counter()
    bm = prime(ns, version, chunks)
    bm.fetch_ms = round((t1 - t0) * 1000, 1)
    log.info("bm25 index %s: %d chunks, fetch %.0f ms, build %.0f ms",
             ns, len(chunks), bm.fetch_ms, bm.build_ms)
    return bm


def prime(ns: str, version: str, chunks: list[dict]) -> BM25:
    """
    Builds and caches the index from chunks already in memory. Ingestion calls
    this with the chunks it just embedded, so the first question after
    ingestion does not have to read them back from Pinecone.
    """
    t0 = time.perf_counter()
    chunks = sorted(chunks, key=lambda c: c["id"])  # same order however it was loaded
    bm = BM25(chunks, config.BM25_K1, config.BM25_B)
    bm.build_ms = round((time.perf_counter() - t0) * 1000, 1)
    bm.fetch_ms = 0.0
    with _lock:
        _cache[ns] = (version, bm)
        _cache.move_to_end(ns)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return bm


def rrf(ranked_lists: list[list[dict]], weights: list[float], k: int, top_k: int) -> list[dict]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
        score(d) = sum over lists of  weight / (k + rank of d in that list)
    Uses only ranks, so a cosine score of 0.82 and a BM25 score of 14.7 never
    have to be put on the same scale. k = 60 damps the gap between rank 1 and
    rank 2, so a chunk ranked well by BOTH methods beats one ranked first by
    only one of them.
    """
    score: dict[str, float] = {}
    best: dict[str, dict] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, c in enumerate(lst, start=1):
            score[c["id"]] = score.get(c["id"], 0.0) + w / (k + rank)
            best.setdefault(c["id"], c)
    order = sorted(score, key=score.get, reverse=True)[:top_k]
    return [{**best[i], "score": score[i]} for i in order]
