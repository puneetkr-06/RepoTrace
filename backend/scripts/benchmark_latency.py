"""
Measures chat latency against the deployed API, as the user experiences it.

    cd backend
    python scripts/benchmark_latency.py --api https://<your-render-app>.onrender.com \
        --repo owner/name --questions scripts/eval_questions.jsonl --n 100

Needs ADMIN_TOKEN (same value as on Render) in backend/.env to skip the
per-IP chat limit. Requests are paced at GEMINI_RPM so the run stays inside
Gemini's per-minute limit: 100 requests at 10 RPM take ~10 minutes and use
100 of the day's GEMINI_RPD requests.

Reports p50 / p95 of:
  first_token  time until the first word appears (what "feels fast" means)
  total        time until the full answer is done

Run it with the Render instance already awake (call /api/health first) and
say so when quoting the number: a cold start adds tens of seconds.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402



def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the value below which p% of samples fall."""
    if not values:
        return None
    s = sorted(values)
    return s[max(1, math.ceil(p / 100 * len(s))) - 1]

def one(api: str, repo: str, question: str) -> tuple[float | None, float, bool]:
    t0 = time.perf_counter()
    first = None
    ok = False
    with requests.post(
        f"{api}/api/chat",
        json={"repo": repo, "message": question, "history": []},
        headers={"x-admin-token": config.ADMIN_TOKEN},
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "token" and first is None:
                first = time.perf_counter() - t0
            if ev["type"] == "done":
                ok = True
    return first, time.perf_counter() - t0, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    qs = [json.loads(l)["question"] for l in Path(args.questions).read_text().splitlines() if l.strip()]
    requests.get(f"{args.api}/api/health", timeout=120)  # wake the instance

    gap = 60 / config.GEMINI_RPM
    firsts, totals, errors = [], [], 0
    for i in range(args.n):
        start = time.perf_counter()
        try:
            f, t, ok = one(args.api, args.repo, qs[i % len(qs)])
            if ok and f is not None:
                firsts.append(f)
                totals.append(t)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            print("error:", e)
        print(f"{i + 1}/{args.n}", end="\r", flush=True)
        time.sleep(max(0.0, gap - (time.perf_counter() - start)))

    print(f"\nsuccessful: {len(totals)}  errors: {errors}")
    for name, vals in (("first_token", firsts), ("total", totals)):
        print(f"{name:12s} p50 {percentile(vals, 50):.2f}s   p95 {percentile(vals, 95):.2f}s")


if __name__ == "__main__":
    main()
