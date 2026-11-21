"""
Language-aware chunking.

RecursiveCharacterTextSplitter.from_language() splits on each language's
syntax separators (class / def / function / fn ...) before falling back to
blank lines and characters. It is NOT an AST parser: a function longer than
CHUNK_SIZE is still split inside its body. Describe it as "language-aware",
not "AST-aware".
"""
import bisect
import hashlib
import os
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import config
from github_source import SourceFile

# 28 extensions -> 20 distinct languages
EXTENSION_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".cpp": Language.CPP, ".hpp": Language.CPP, ".cu": Language.CPP, ".cuh": Language.CPP,
    ".c": Language.C, ".h": Language.C,
    ".cs": Language.CSHARP,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".sol": Language.SOL,
    ".ex": Language.ELIXIR, ".exs": Language.ELIXIR,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
}

_splitters: dict = {}


def _splitter(ext: str) -> RecursiveCharacterTextSplitter:
    lang = EXTENSION_LANGUAGE.get(ext)
    key = lang or "generic"
    if key not in _splitters:
        kwargs = dict(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            add_start_index=True,  # gives each chunk's character offset in the file
        )
        _splitters[key] = (
            RecursiveCharacterTextSplitter.from_language(language=lang, **kwargs)
            if lang
            else RecursiveCharacterTextSplitter(**kwargs)
        )
    return _splitters[key]


@dataclass
class Chunk:
    id: str
    path: str
    language: str
    start_line: int
    end_line: int
    code: str

    @property
    def embed_text(self) -> str:
        # The file path goes into the embedded text so that questions like
        # "where is auth handled?" can match on file and folder names too.
        return f"File: {self.path}\n\n{self.code}"


def _line_of(newlines: list[int], offset: int) -> int:
    """1-based line number of a character offset."""
    return bisect.bisect_right(newlines, offset - 1) + 1


def chunk_file(f: SourceFile) -> list[Chunk]:
    ext = os.path.splitext(f.path)[1].lower()
    lang = EXTENSION_LANGUAGE.get(ext)
    newlines = [i for i, ch in enumerate(f.text) if ch == "\n"]
    docs = _splitter(ext).create_documents([f.text])
    chunks = []
    for i, d in enumerate(docs):
        start = d.metadata.get("start_index", 0)
        if start < 0:  # splitter could not locate the chunk; fall back to file start
            start = 0
        code = d.page_content
        start_line = _line_of(newlines, start)
        end_line = start_line + code.count("\n")
        chunks.append(
            Chunk(
                # Pinecone IDs are kept ASCII and short: hash of the path + index.
                id=f"{hashlib.sha1(f.path.encode()).hexdigest()[:16]}-{i}",
                path=f.path,
                language=lang.value if lang else "text",
                start_line=start_line,
                end_line=end_line,
                code=code,
            )
        )
    return chunks


def chunk_files(files: list[SourceFile]) -> list[Chunk]:
    out: list[Chunk] = []
    for f in files:
        out.extend(chunk_file(f))
    return out


def estimate_tokens(chunks: list[Chunk]) -> int:
    """Upper-side estimate used for the budget check before spending tokens."""
    chars = sum(len(c.embed_text) for c in chunks)
    return int(chars / config.CHARS_PER_TOKEN_EST)
