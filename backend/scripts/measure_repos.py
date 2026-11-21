"""
Measures the [MEASURE] numbers in config.py on real repositories.

    cd backend
    python scripts/measure_repos.py facebook/react tiangolo/fastapi gin-gonic/gin

Per repo it reports:
  * file-size distribution (p50 / p95 / p99)       -> MAX_FILE_BYTES
  * characters per token on a random chunk sample  -> CHARS_PER_TOKEN_EST
  * share of chunks over the 507-token model limit -> is CHUNK_SIZE safe?
  * tarball size and estimated tokens for the repo -> MAX_TARBALL_BYTES, MAX_REPO_TOKENS

Cost: each sampled chunk is embedded once (~300-450 tokens), so --sample 100
uses roughly 40K of the 5M monthly Pinecone tokens per repo.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# See every file, including the ones the production limit would skip.
config.MAX_FILE_BYTES = 10**9

import chunking  # noqa: E402
import vector_store  # noqa: E402
from github_source import get_repo_info, load_files  # noqa: E402
from metrics import percentile  # noqa: E402


def tokens_of(text: str) -> int | None:
    """Exact token count; None if the text is over the model limit."""
    try:
        res = vector_store._client().inference.embed(
            model=config.EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "passage", "truncate": "NONE"},
        )
    except Exception as e:
        if "429" in str(e):
            raise
        return None  # rejected because it is longer than 507 tokens
    return int(vector_store._get(vector_store._get(res, "usage"), "total_tokens", 0))


def measure(repo: str, sample: int, rng: random.Random) -> dict:
    info = get_repo_info(repo)
    files, stats = load_files(info)
    chunks = chunking.chunk_files(files)
    picked = rng.sample(chunks, min(sample, len(chunks)))

    cpt, over_limit = [], 0
    for c in picked:
        t = tokens_of(c.embed_text)
        if t is None:
            over_limit += 1
        elif t:
            cpt.append(len(c.embed_text) / t)

    embed_chars = sum(len(c.embed_text) for c in chunks)
    c50 = percentile(cpt, 50)
    c5 = percentile(cpt, 5)
    return {
        "repo": info.full_name,
        "tarball_mb": round(stats.tarball_bytes / 1e6, 2),
        "files": stats.files_kept,
        "file_bytes_p50": percentile(stats.file_sizes, 50),
        "file_bytes_p95": percentile(stats.file_sizes, 95),
        "file_bytes_p99": percentile(stats.file_sizes, 99),
        "chunks": len(chunks),
        "embed_chars": embed_chars,
        "sampled_chunks": len(picked),
        "chars_per_token_p5": round(c5, 3) if c5 else None,
        "chars_per_token_p50": round(c50, 3) if c50 else None,
        "pct_chunks_over_507_tokens": round(100 * over_limit / max(1, len(picked)), 1),
        "est_repo_tokens_p50": int(embed_chars / c50) if c50 else None,
        "est_repo_tokens_worst": int(embed_chars / c5) if c5 else None,
        "_file_sizes": stats.file_sizes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default="measurements.json")
    args = ap.parse_args()

    rng = random.Random(42)  # fixed seed -> same sample on re-runs
    results = []
    for repo in args.repos:
        print(f"measuring {repo} ...", flush=True)
        r = measure(repo, args.sample, rng)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, indent=2))

    all_sizes = [s for r in results for s in r["_file_sizes"]]
    c5s = [r["chars_per_token_p5"] for r in results if r["chars_per_token_p5"]]
    c5 = min(c5s) if c5s else None
    rec = {
        "CHARS_PER_TOKEN_EST (lowest per-repo p5)": c5,
        "MAX_FILE_BYTES (p99 of all files)": percentile(all_sizes, 99),
        "max safe CHUNK_SIZE = 507 x c5 x 0.9 - 40 (header)":
            math.floor(507 * c5 * 0.9 - 40) if c5 else None,
        "MAX_TARBALL_BYTES (largest seen x 2)":
            int(max(r["tarball_mb"] for r in results) * 2 * 1e6),
    }
    print("\nRECOMMENDED SETTINGS")
    print(json.dumps(rec, indent=2))

    for r in results:
        r.pop("_file_sizes")
    Path(args.out).write_text(json.dumps({"repos": results, "recommended": rec}, indent=2))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
