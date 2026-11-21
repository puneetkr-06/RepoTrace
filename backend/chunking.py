"""
Code chunking. Two chunkers:

  "ast"       tree-sitter parses each file into a syntax tree. Chunks are cut
              only at node boundaries: whole functions, classes and statements
              are packed together up to CHUNK_SIZE; a node larger than that is
              split into its children, recursively. Comments and blank lines
              before a node stay with that node. No overlap is needed, because
              boundaries already fall between logical units.
              Used for every supported extension whose grammar package is
              installed (see _GRAMMARS; ast_languages() lists them at runtime).

  "splitter"  LangChain's RecursiveCharacterTextSplitter.from_language(): splits
              on keyword patterns (\\nclass, \\ndef, \\nfunction...), then blank
              lines, then characters. It never parses the code, so a long
              function can be cut mid-body. Used for every other language, and
              for everything when tree-sitter is not installed.

integrity() measures how many function/class definitions end up whole inside
one chunk, so the two chunkers can be compared with a number.
"""
import bisect
import hashlib
import importlib
import logging
import os
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import config
from github_source import SourceFile

log = logging.getLogger("repotrace.chunking")

# 28 extensions -> 20 distinct languages (used by the splitter, and for labels)
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

# ------------------------------------------------------------------ tree-sitter
# extension -> (grammar package, function returning the language pointer)
_GRAMMARS = {
    ".py": ("tree_sitter_python", "language"),
    ".js": ("tree_sitter_javascript", "language"),
    ".jsx": ("tree_sitter_javascript", "language"),
    ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".c": ("tree_sitter_c", "language"),
    ".h": ("tree_sitter_c", "language"),
    # CUDA is C++ plus a few extensions (__global__, <<<...>>>); the C++ grammar
    # marks those few spots as errors but still parses the rest of the file.
    ".cpp": ("tree_sitter_cpp", "language"),
    ".hpp": ("tree_sitter_cpp", "language"),
    ".cu": ("tree_sitter_cpp", "language"),
    ".cuh": ("tree_sitter_cpp", "language"),
    ".java": ("tree_sitter_java", "language"),
    ".go": ("tree_sitter_go", "language"),
    ".rs": ("tree_sitter_rust", "language"),
    # The rest of the supported extensions. Each grammar is optional: if its
    # package is not installed, that extension falls back to the splitter.
    ".html": ("tree_sitter_html", "language"),
    ".css": ("tree_sitter_css", "language"),
    ".md": ("tree_sitter_markdown", "language"),
    ".cs": ("tree_sitter_c_sharp", "language"),
    ".rb": ("tree_sitter_ruby", "language"),
    ".php": ("tree_sitter_php", "language_php"),
    ".swift": ("tree_sitter_swift", "language"),
    ".kt": ("tree_sitter_kotlin", "language"),
    ".scala": ("tree_sitter_scala", "language"),
    ".sol": ("tree_sitter_solidity", "language"),
    ".ex": ("tree_sitter_elixir", "language"),
    ".exs": ("tree_sitter_elixir", "language"),
    ".lua": ("tree_sitter_lua", "language"),
    ".pl": ("tree_sitter_perl", "language"),
}

# Node types counted as "definitions" by integrity().
DEFINITION_TYPES = {
    # Python
    "function_definition", "class_definition", "decorated_definition",
    # JavaScript / TypeScript
    "function_declaration", "generator_function_declaration", "class_declaration",
    "method_definition", "arrow_function", "function_expression", "interface_declaration",
    # C / C++
    "class_specifier", "struct_specifier",
    # Java
    "method_declaration", "constructor_declaration",
    # Rust
    "function_item", "impl_item",
    # C#, Kotlin, Swift, Scala
    "object_declaration", "protocol_declaration", "object_definition", "trait_definition",
    # Ruby
    "method", "singleton_method", "class", "module",
    # Solidity, Perl
    "contract_declaration", "modifier_definition", "subroutine_declaration_statement",
}


def _load_grammars():
    try:
        from tree_sitter import Language as TSLanguage, Parser
    except ImportError:
        log.warning("tree-sitter not installed: using the keyword splitter for all files")
        return None, {}
    langs, cache = {}, {}
    for ext, (module, fn) in _GRAMMARS.items():
        key = (module, fn)
        if key not in cache:
            try:
                cache[key] = TSLanguage(getattr(importlib.import_module(module), fn)())
            except Exception as e:  # missing or incompatible grammar package
                log.warning("tree-sitter grammar %s unavailable: %s", module, e)
                cache[key] = None
        if cache[key] is not None:
            langs[ext] = cache[key]
    return Parser, langs


_Parser, _TS_LANGS = _load_grammars()
AST_AVAILABLE = bool(_TS_LANGS)


def ast_languages() -> list[str]:
    """Extensions that are chunked with tree-sitter in this process."""
    return sorted(_TS_LANGS)


def active_chunker() -> str:
    return "ast" if config.CHUNKER == "ast" and AST_AVAILABLE else "splitter"


def _parse(ext: str, src: bytes):
    # A new Parser per call: Parser objects are not safe to share across threads.
    return _Parser(_TS_LANGS[ext]).parse(src)


# ------------------------------------------------------------------ chunk model
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


def _chunk_id(path: str, i: int) -> str:
    # Pinecone IDs are kept ASCII and short: hash of the path + index.
    return f"{hashlib.sha1(path.encode()).hexdigest()[:16]}-{i}"


def _label(ext: str) -> str:
    lang = EXTENSION_LANGUAGE.get(ext)
    return lang.value if lang else "text"


# ------------------------------------------------------------------ splitter chunker
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


def _line_of(newlines: list[int], offset: int) -> int:
    """1-based line number of a character offset."""
    return bisect.bisect_right(newlines, offset - 1) + 1


def _chunk_with_splitter(f: SourceFile, ext: str) -> list[Chunk]:
    newlines = [i for i, ch in enumerate(f.text) if ch == "\n"]
    chunks = []
    for i, d in enumerate(_splitter(ext).create_documents([f.text])):
        start = max(d.metadata.get("start_index", 0), 0)
        code = d.page_content
        start_line = _line_of(newlines, start)
        chunks.append(Chunk(_chunk_id(f.path, i), f.path, _label(ext),
                            start_line, start_line + code.count("\n"), code))
    return chunks


# ------------------------------------------------------------------ AST chunker
def _split_leaf(lo: int, hi: int, src: bytes, out: list):
    """A single node bigger than CHUNK_SIZE with no children (a huge string):
    cut at line ends."""
    limit = config.CHUNK_SIZE
    while hi - lo > limit:
        cut = src.rfind(b"\n", lo, lo + limit)
        cut = lo + limit if cut <= lo else cut + 1
        out.append((lo, cut))
        lo = cut
    if hi > lo:
        out.append((lo, hi))


def _units(node, lo: int, hi: int, src: bytes, out: list):
    """
    Cover [lo, hi) with contiguous byte spans, each one AST node (plus the
    comments / whitespace just before it) no bigger than CHUNK_SIZE.
    A child that is too big is replaced by its own children, recursively.
    """
    kids = node.children
    if not kids:
        _split_leaf(lo, hi, src, out)
        return
    bounds = [lo] + [k.end_byte for k in kids[:-1]] + [hi]
    for k, s, e in zip(kids, bounds, bounds[1:]):
        if e <= s:
            continue
        if e - s <= config.CHUNK_SIZE:
            out.append((s, e))
            continue
        node_start = max(s, k.start_byte)
        if node_start > s and e - node_start <= config.CHUNK_SIZE:
            # The node fits; only the comments / blank lines before it push the
            # unit over the limit. Cut that gap off at line ends and keep the
            # node whole, instead of splitting a node that would have fitted.
            _split_leaf(s, node_start, src, out)
            out.append((node_start, e))
        else:
            _units(k, s, e, src, out)


def _chunk_with_ast(f: SourceFile, ext: str) -> list[Chunk]:
    src = f.text.encode("utf-8")
    tree = _parse(ext, src)
    units: list = []
    _units(tree.root_node, 0, len(src), src, units)

    # Pack neighbouring units together while the chunk stays <= CHUNK_SIZE.
    spans, cs, ce = [], None, None
    for s, e in units:
        if cs is None:
            cs, ce = s, e
        elif e - cs <= config.CHUNK_SIZE:
            ce = e
        else:
            spans.append((cs, ce))
            cs, ce = s, e
    if cs is not None:
        spans.append((cs, ce))

    chunks = []
    for s, e in spans:
        raw = src[s:e].decode("utf-8", "replace")
        code = raw.lstrip("\r\n")
        start_line = src.count(b"\n", 0, s) + 1 + raw[: len(raw) - len(code)].count("\n")
        code = code.rstrip()
        if not code:
            continue
        chunks.append(Chunk(_chunk_id(f.path, len(chunks)), f.path, _label(ext),
                            start_line, start_line + code.count("\n"), code))
    return chunks


# ------------------------------------------------------------------ public API
def chunk_file(f: SourceFile, chunker: str | None = None) -> list[Chunk]:
    chunker = chunker or active_chunker()
    ext = os.path.splitext(f.path)[1].lower()
    if chunker == "ast" and ext in _TS_LANGS:
        try:
            return _chunk_with_ast(f, ext)
        except Exception:
            log.exception("AST chunking failed for %s; using the splitter", f.path)
    return _chunk_with_splitter(f, ext)


def chunk_files(files: list[SourceFile], chunker: str | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for f in files:
        out.extend(chunk_file(f, chunker))
    return out


def estimate_tokens(chunks: list[Chunk]) -> int:
    """Upper-side estimate used for the budget check before spending tokens."""
    chars = sum(len(c.embed_text) for c in chunks)
    return int(chars / config.CHARS_PER_TOKEN_EST)


def integrity(files: list[SourceFile], chunks: list[Chunk]) -> dict:
    """
    Of the function/class definitions small enough to fit in one chunk
    (<= CHUNK_SIZE bytes), how many lie entirely inside a single chunk?
    Needs tree-sitter (it finds the definitions); covers the languages above.
    """
    if not AST_AVAILABLE:
        return {"definitions": None, "fit": None, "intact": None, "intact_pct": None, "broken": []}
    by_path: dict[str, list] = {}
    for c in chunks:
        by_path.setdefault(c.path, []).append((c.start_line, c.end_line))
    total = fit = intact = 0
    broken: list = []  # definitions that fit but were split (first 20)
    for f in files:
        ext = os.path.splitext(f.path)[1].lower()
        if ext not in _TS_LANGS:
            continue
        src = f.text.encode("utf-8")
        ranges = by_path.get(f.path, [])
        stack = [_parse(ext, src).root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            # is_named: keyword tokens such as `class` in JavaScript share
            # a type name with real nodes in other grammars; skip them.
            # n.parent is None: the file's root node (Python calls it
            # "module", the same name as a Ruby module); not a definition.
            if n.parent is None or not n.is_named or n.type not in DEFINITION_TYPES:
                continue
            total += 1
            if n.end_byte - n.start_byte > config.CHUNK_SIZE:
                continue  # cannot fit in one chunk with either chunker
            fit += 1
            a, b = n.start_point[0] + 1, n.end_point[0] + 1
            if any(s <= a and b <= e for s, e in ranges):
                intact += 1
            elif len(broken) < 20:
                broken.append({"path": f.path, "type": n.type, "lines": [a, b],
                               "bytes": n.end_byte - n.start_byte})
    return {
        "definitions": total,
        "fit": fit,
        "intact": intact,
        "intact_pct": round(100 * intact / fit, 1) if fit else None,
        "broken": broken,
    }
