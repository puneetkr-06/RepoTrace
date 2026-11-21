"""
Retrieval evaluation: recall@k with a 95% confidence interval.

    cd backend
    python scripts/evaluate_retrieval.py scripts/eval_questions.jsonl [--mode dense|bm25|hybrid]

Input: one JSON object per line
    {"repo": "owner/name", "question": "...", "expected_files": ["path/a.py"]}
The repos must already be ingested.

recall@k = share of questions where at least one expected file appears in the
top-k retrieved chunks.

95% CI = +/- 1.96 * sqrt(p * (1 - p) / n). With p = 0.8:
    n = 30 -> +/-14.3 pts,  n = 100 -> +/-7.8 pts,  n = 200 -> +/-5.5 pts
so use at least ~100 questions before quoting a number.

Choosing TOP_K: the smallest k after which each extra chunk adds less than
2 recall points. More chunks past that point mostly add prompt tokens.

Cost: one query embedding (~30 tokens) and one Pinecone query per question.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
import vector_store  # noqa: E402
from github_source import parse_repo  # noqa: E402

MAX_K = 10
GAIN_THRESHOLD = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--mode", default=None, help="dense | bm25 | hybrid (default: config)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    first_hit_rank = []  # 1-based rank of the first relevant chunk, or None
    misses = []
    for row in rows:
        repo = parse_repo(row["repo"])
        version = rag.index_version(vector_store.read_meta(repo) or {})
        chunks, _ = rag.retrieve(repo, row["question"], [], top_k=MAX_K, version=version, mode=args.mode)
        expected = set(row.get("expected_files") or [e["path"] for e in row["expected"]])
        rank = next((i + 1 for i, c in enumerate(chunks) if c["path"] in expected), None)
        first_hit_rank.append(rank)
        if rank is None:
            misses.append({"question": row["question"], "got": [c["path"] for c in chunks[:3]]})

    n = len(rows)
    print(f"questions: {n}\n")
    print(" k  recall   95% CI")
    recalls = []
    for k in range(1, MAX_K + 1):
        p = sum(1 for r in first_hit_rank if r is not None and r <= k) / n
        half = 1.96 * math.sqrt(p * (1 - p) / n)
        recalls.append(p)
        print(f"{k:2d}  {p:6.1%}   +/-{half:5.1%}")

    mrr = sum(1 / r for r in first_hit_rank if r) / n
    print(f"\nMRR@{MAX_K}: {mrr:.3f}")

    suggested = MAX_K
    for k in range(1, MAX_K):
        if all(recalls[j] - recalls[j - 1] < GAIN_THRESHOLD for j in range(k, MAX_K)):
            suggested = k
            break
    print(f"suggested TOP_K (each extra chunk adds < {GAIN_THRESHOLD:.0%}): {suggested}")

    if misses:
        print(f"\n{len(misses)} questions with no relevant file in top {MAX_K}:")
        for m in misses[:10]:
            print(" -", m["question"], "->", m["got"])


if __name__ == "__main__":
    main()
