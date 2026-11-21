"""
Downloads a public GitHub repository as ONE tarball request and yields its
source files. The old version used GithubFileLoader, which makes one API call
per file: a 2,000-file repo used ~2,000 of the 5,000 requests/hour that a
token gets. A tarball is 1 request no matter how many files the repo has.
"""
import re
import tarfile
import tempfile
from dataclasses import dataclass, field

import requests

import config

API = "https://api.github.com"
TIMEOUT = 15  # seconds per metadata call

SOURCE_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".html", ".css",
    ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".cs", ".java", ".rs", ".go",
    ".rb", ".php", ".swift", ".kt", ".scala", ".sol", ".ex", ".exs", ".lua", ".pl",
)
SKIP_DIRS = {
    "node_modules", "dist", "build", "vendor", ".git", "__pycache__", "venv",
    ".venv", ".next", "target", "coverage", "third_party",
    # Build-tool caches. Found while measuring: one repo had 50K chars of
    # committed Vite pre-bundled deps in frontend/.vite/deps.
    ".vite", ".cache", ".parcel-cache", ".turbo", ".svelte-kit",
}
SKIP_SUFFIXES = (".min.js", ".min.css", ".map", ".lock", "package-lock.json")

_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_NAME = r"[A-Za-z0-9._-]{1,100}"
_REPO_RE = re.compile(
    rf"^(?:https?://)?(?:www\.)?(?:github\.com/)?({_OWNER})/({_NAME}?)(?:\.git)?(?:/.*)?$"
)


class RepoError(Exception):
    """A problem the user can fix (bad URL, private repo, too big...)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def parse_repo(text: str) -> str:
    """'https://github.com/a/b.git', 'github.com/a/b/tree/main', 'a/b' -> 'a/b'."""
    m = _REPO_RE.match(text.strip().rstrip("/"))
    if not m or not m.group(2):
        raise RepoError("Invalid GitHub repository. Use owner/repo or a github.com URL.")
    name = m.group(2)
    if name.endswith(".git"):
        name = name[:-4]
    return f"{m.group(1)}/{name}"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return h


@dataclass
class RepoInfo:
    full_name: str       # canonical casing from GitHub, used as the Pinecone namespace
    default_branch: str
    commit_sha: str      # exact commit we ingest; citations link to it


def get_repo_info(repo: str) -> RepoInfo:
    r = requests.get(f"{API}/repos/{repo}", headers=_headers(), timeout=TIMEOUT)
    if r.status_code == 401:
        # The server's GITHUB_TOKEN is expired or revoked: an operator problem.
        raise RepoError("Server GitHub token is invalid or expired.", 502)
    if r.status_code == 404:
        raise RepoError(f"Repository '{repo}' not found or not public.", 404)
    if r.status_code == 403:
        raise RepoError("GitHub rate limit reached. Try again later.", 429)
    r.raise_for_status()
    data = r.json()
    # The server token may have access to private repos. Never ingest those:
    # anyone could then read them through the chat.
    if data.get("private"):
        raise RepoError("Only public repositories are supported.", 403)

    branch = data.get("default_branch") or "main"
    full_name = data["full_name"]
    c = requests.get(
        f"{API}/repos/{full_name}/commits/{branch}",
        headers={**_headers(), "Accept": "application/vnd.github.sha"},
        timeout=TIMEOUT,
    )
    c.raise_for_status()
    return RepoInfo(full_name=full_name, default_branch=branch, commit_sha=c.text.strip())


@dataclass
class SourceFile:
    path: str
    text: str


@dataclass
class LoadStats:
    files_kept: int = 0
    files_skipped_type: int = 0
    files_skipped_size: int = 0
    files_skipped_binary: int = 0
    chars: int = 0
    tarball_bytes: int = 0
    file_sizes: list = field(default_factory=list)  # bytes of every kept file


def _is_wanted(path: str) -> bool:
    parts = path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    name = parts[-1]
    return name.endswith(SOURCE_EXTENSIONS) and not name.endswith(SKIP_SUFFIXES)


def load_files(info: RepoInfo, max_chars: int | None = None):
    """
    Returns (files, stats). Raises RepoError if the repo is over the size limits.
    max_chars: stop early once raw text exceeds this (repo too big for budget).
    """
    stats = LoadStats()
    url = f"{API}/repos/{info.full_name}/tarball/{info.commit_sha}"
    files: list[SourceFile] = []

    with tempfile.TemporaryFile() as tmp:
        with requests.get(url, headers=_headers(), stream=True, timeout=60) as r:
            r.raise_for_status()
            for block in r.iter_content(chunk_size=1 << 20):
                stats.tarball_bytes += len(block)
                if stats.tarball_bytes > config.MAX_TARBALL_BYTES:
                    raise RepoError(
                        f"Repository download is over {config.MAX_TARBALL_BYTES // 1_000_000} MB."
                    )
                tmp.write(block)
        tmp.seek(0)

        # Members are read in memory with extractfile(); nothing is written to
        # disk, so paths like '../../etc' in a malicious tarball are harmless.
        with tarfile.open(fileobj=tmp, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # Tarball paths start with "<owner>-<repo>-<sha>/"; drop that.
                path = member.name.split("/", 1)[-1]
                if not _is_wanted(path):
                    stats.files_skipped_type += 1
                    continue
                if member.size > config.MAX_FILE_BYTES:
                    stats.files_skipped_size += 1
                    continue
                raw = tar.extractfile(member).read()
                if b"\x00" in raw[:8192]:
                    stats.files_skipped_binary += 1
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    stats.files_skipped_binary += 1
                    continue
                if not text.strip():
                    continue
                files.append(SourceFile(path=path, text=text))
                stats.files_kept += 1
                stats.chars += len(text)
                stats.file_sizes.append(len(raw))
                if max_chars is not None and stats.chars > max_chars:
                    raise RepoError(
                        "Repository is too large for the free-tier token budget "
                        f"(over {max_chars:,} characters of source).",
                        413,
                    )
    return files, stats
