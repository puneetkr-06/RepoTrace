"""
All tunable numbers live here, read from environment variables.

Every default has a reason written next to it. Three kinds of numbers:
  [DOC]      a documented limit of an external service
  [DERIVED]  calculated from [DOC] numbers with the formula shown
  [MEASURE]  a starting value that must be replaced by a measured one
             (see backend/scripts/measure_repos.py and evaluate_retrieval.py)
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------- secrets
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# Lets backend/scripts/benchmark_latency.py skip the per-IP limits.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# Comma-separated list, e.g. "https://repotrace.vercel.app,http://localhost:5173"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

# ---------------------------------------------------------------- Pinecone
INDEX_NAME = os.environ.get("PINECONE_INDEX", "repotrace-pc")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"  # [DOC] Starter plan indexes run on AWS us-east-1

EMBED_MODEL = "multilingual-e5-large"
EMBED_DIM = 1024               # [DOC] output dimension of multilingual-e5-large
EMBED_MAX_TOKENS = 507         # [DOC] max input tokens per text
EMBED_BATCH = 96               # [DOC] max texts per embed request
EMBED_TPM = _int("EMBED_TPM", 250_000)  # [DOC] Starter plan tokens/minute

# [DOC] Starter plan: 5M embedding tokens per month.
MONTHLY_EMBED_TOKENS = _int("MONTHLY_EMBED_TOKENS", 5_000_000)
# [DERIVED] Kept aside for query embeddings (every chat embeds the question).
# 500K = 10% of the month. At ~30 tokens per question [MEASURE] that is ~16K chats.
QUERY_TOKEN_RESERVE = _int("QUERY_TOKEN_RESERVE", 500_000)

# Upsert batch: one record is 1024 floats (~10 KB as JSON) + ~1 KB metadata.
# 96 x ~11 KB = ~1 MB per request, under Pinecone's 2 MB request limit [DOC].
UPSERT_BATCH = 96

# ---------------------------------------------------------------- chunking
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)
# "ast" = tree-sitter chunking (cuts only at syntax-node boundaries) for the
# languages it has grammars for; "splitter" = LangChain keyword splitter.
# Changing it makes every repo re-index on its next request.
CHUNKER = os.environ.get("CHUNKER", "ast")
# Each chunk adds (CHUNK_SIZE - CHUNK_OVERLAP) new characters, so overlap costs
# CHUNK_SIZE / (CHUNK_SIZE - CHUNK_OVERLAP) = 1000 / 800 = 1.25x embedding tokens.
OVERLAP_FACTOR = CHUNK_SIZE / (CHUNK_SIZE - CHUNK_OVERLAP)

# Characters per token for code with the e5 tokenizer. [MEASURE]
# 2.5 is deliberately pessimistic (a lower value over-estimates tokens, so the
# budget check fails safe). Replace with the p5 value from measure_repos.py.
CHARS_PER_TOKEN_EST = _float("CHARS_PER_TOKEN_EST", 2.5)
# Sanity check: 1000-char chunk + ~40-char "File: path" header at 2.5 chars/token
# = ~416 tokens, below the 507-token model limit [DOC].

# ---------------------------------------------------------------- repo limits
# [DERIVED] Per-repo token cap = (monthly tokens - query reserve) / N,
# where N = new repos we want to be able to ingest per month.
# N = 18  ->  (5,000,000 - 500,000) / 18 = 250,000 tokens per repo.
MAX_REPO_TOKENS = _int("MAX_REPO_TOKENS", 250_000)

# [DERIVED] One file at 100 KB = 100,000 / 2.5 = 40,000 tokens = 16% of a
# 250K repo budget. Files that large are almost always generated, minified or
# vendored code. Replace with the measured p99 source-file size. [MEASURE]
MAX_FILE_BYTES = _int("MAX_FILE_BYTES", 100_000)

# Safety guard on the download (it goes to a temp file, not RAM).
# Tune with the tarball sizes reported by measure_repos.py. [MEASURE]
MAX_TARBALL_BYTES = _int("MAX_TARBALL_BYTES", 100_000_000)

# ---------------------------------------------------------------- retrieval / LLM
# [MEASURED] Picked on the 57 dev questions with evaluate_retrieval.py: smallest
# k after which every extra chunk adds < 2 points of line-level recall.
# Dev line recall: k=6 84.2%, k=7 87.7%, k=8 91.2%, k=9 93.0%, k=10 93.0%
# (gains 3.5, 1.8, 0.0 after k=7) -> 8. Held-out test set at k=8: 85.6% +/- 7.3.
# Re-checked with hybrid retrieval: dev curve k=6 91.2%, k=7..10 96.5%. The
# rule now says 7, but dev is at its ceiling (2 misses left; one question is
# 1.75 points, under the 2-point step), so it cannot separate 7 from 8. Kept 8,
# which was fixed before the hybrid test run. Test at k=8: 93.3% +/- 5.2.
TOP_K = _int("TOP_K", 8)
# "hybrid" = vector + BM25 keyword search fused with RRF; "dense" = vector only
# (the previous behaviour); "bm25" = keyword only. Used for A/B evaluation.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")
# [LITERATURE] Okapi BM25 defaults (Robertson & Zaragoza 2009; Lucene/Elastic
# use the same): k1 = how fast repeated words stop adding score, b = how much
# long chunks are penalised.
BM25_K1 = _float("BM25_K1", 1.2)
BM25_B = _float("BM25_B", 0.75)
# [LITERATURE] RRF constant from Cormack et al. 2009, the standard default.
RRF_K = _int("RRF_K", 60)
# [MEASURED on dev only] How deep each list is read before fusing, and the
# weight of the keyword list. Dev line recall@8 for N = 10/20/30/50 at w = 1:
# 94.7 / 94.7 / 96.5 / 98.2; for w = 0.5/0.75/1/1.5/2 at N = 30:
# 94.7 / 96.5 / 96.5 / 96.5 / 96.5 (MRR best at w = 1). RRF k = 10..100 gave
# the same recall. N = 30 is the smallest N within 2 points of the best.
HYBRID_CANDIDATES = _int("HYBRID_CANDIDATES", 30)
# [MEASURED] BM25 indexes kept in memory, counted in chunks. ~8 KB per chunk
# (tracemalloc), so 5,000 chunks = ~40 MB: about 45 average repos or 5 repos at
# the token cap. An evicted repo is rebuilt from Pinecone in under a second.
BM25_CACHE_MAX_CHUNKS = _int("BM25_CACHE_MAX_CHUNKS", 5000)
KEYWORD_WEIGHT = _float("KEYWORD_WEIGHT", 1.0)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
# Low temperature because answers must stick to retrieved code. No formula for
# this one: compare 0 and 0.2 with the eval set before changing it.
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2)
# Measured locally: first token at 4.2-5.5 s and full answer at ~4.8-5.5 s, i.e.
# almost all the wait happens BEFORE the first word. That is Gemini 2.5 Flash
# "thinking". 0 turns thinking off for 2.5 Flash. Re-check answer quality with
# the eval set after changing it; set -1 to let the model decide again.
LLM_THINKING_BUDGET = _int("LLM_THINKING_BUDGET", 0)
# A question this short that also contains "it/this/that/more..." is treated
# as a follow-up and searched together with the previous question.
# [MEASURE] tune with follow-up questions in the eval set.
FOLLOWUP_MAX_WORDS = _int("FOLLOWUP_MAX_WORDS", 8)
# Retrieved context is about TOP_K x ~1040 chars = ~6,200 chars. History is kept
# below that so retrieved code, not old chat, dominates the prompt.
MAX_HISTORY_CHARS = _int("MAX_HISTORY_CHARS", 4000)

# ---------------------------------------------------------------- rate limits
# [DOC] Gemini free-tier limits change; read the current values for your
# project in Google AI Studio and set them here.
GEMINI_RPM = _int("GEMINI_RPM", 10)
GEMINI_RPD = _int("GEMINI_RPD", 250)
# How many different people per day the demo should support. [MEASURE] from logs.
EXPECTED_DAILY_USERS = _int("EXPECTED_DAILY_USERS", 25)
# [DERIVED] chats per IP per day = GEMINI_RPD / EXPECTED_DAILY_USERS = 250 / 25 = 10
CHAT_PER_IP_PER_DAY = max(1, GEMINI_RPD // EXPECTED_DAILY_USERS)

# [DERIVED] New ingestions per month N = (5M - 0.5M) / 250K = 18.
# One visitor may use at most ~10% of that per day: 18 x 0.1 = ~2.
INGEST_PER_IP_PER_DAY = _int("INGEST_PER_IP_PER_DAY", 2)
