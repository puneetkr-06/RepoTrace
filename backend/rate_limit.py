"""
Sliding-window rate limiter kept in memory (no Redis needed for one instance).
Counters reset if the Render instance restarts; the monthly token budget is
stored in Pinecone and does not reset.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

import config


class SlidingWindow:
    def __init__(self, limit: int, window_s: int):
        self.limit = limit
        self.window = window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = Lock()

    def check(self, key: str) -> int | None:
        """Records a hit. Returns None if allowed, else seconds until a slot frees."""
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return int(self.window - (now - q[0])) + 1
            q.append(now)
            return None


DAY = 24 * 3600

chat_per_ip = SlidingWindow(config.CHAT_PER_IP_PER_DAY, DAY)
ingest_per_ip = SlidingWindow(config.INGEST_PER_IP_PER_DAY, DAY)
# Global: Gemini counts requests per project, not per user.
chat_global_minute = SlidingWindow(config.GEMINI_RPM, 60)
chat_global_day = SlidingWindow(config.GEMINI_RPD, DAY)


def client_ip(request: Request) -> str:
    # Render sits behind a proxy; the real client is the first X-Forwarded-For entry.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_admin(request: Request) -> bool:
    return bool(config.ADMIN_TOKEN) and request.headers.get("x-admin-token") == config.ADMIN_TOKEN


def enforce(limiter: SlidingWindow, key: str, what: str):
    retry = limiter.check(key)
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached for {what}. Try again in {retry} s.",
            headers={"Retry-After": str(retry)},
        )
