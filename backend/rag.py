"""
Retrieval-augmented answering.

Retrieval runs ONCE per question; the same chunks feed both the prompt and the
citations. (The old code retrieved twice: once inside the LCEL chain and once
more to build citations, doubling Pinecone reads and query embeddings.)
"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
import vector_store

SYSTEM_PROMPT = (
    "You are an expert assistant for the GitHub repository {repo}. "
    "Answer ONLY from the retrieved code below. If the code does not contain the "
    "answer, say you don't know. Be concise: 2-3 lines for specific questions; "
    "go into detail only when asked. Use code snippets where helpful and mention "
    "file paths when you refer to code.\n\n"
    "Retrieved code:\n{context}"
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GEMINI_API_KEY,
            thinking_budget=config.LLM_THINKING_BUDGET,
        )
        _chain = _prompt | llm | StrOutputParser()  # LCEL pipeline
    return _chain


def trim_history(history: list[dict]) -> list:
    """Keeps the most recent messages that fit in MAX_HISTORY_CHARS."""
    kept, used = [], 0
    for m in reversed(history):
        content = str(m.get("content", ""))
        if used + len(content) > config.MAX_HISTORY_CHARS:
            break
        used += len(content)
        kept.append(HumanMessage(content) if m.get("role") == "user" else AIMessage(content))
    return list(reversed(kept))


_REFERRING_WORDS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "there", "here", "above", "same", "more", "further", "again",
}


def is_follow_up(question: str) -> bool:
    """
    A follow-up refers back to the previous answer ("explain it more",
    "where is that defined?"). Both conditions must hold:
      * short: <= FOLLOWUP_MAX_WORDS words, since a long question carries its
        own search terms
      * contains a referring word (it / this / that / more ...)
    """
    words = [w.strip("?.!,:;'\"()").lower() for w in question.split()]
    return len(words) <= config.FOLLOWUP_MAX_WORDS and any(w in _REFERRING_WORDS for w in words)


def retrieval_query(question: str, history: list[dict]) -> str:
    """
    Search text for Pinecone. The previous user question is added ONLY for
    follow-ups. Found in local testing: always adding it made "Which file
    extensions are supported?" (asked after a Pinecone question) retrieve
    Pinecone code and answer "not specified"; searched alone it found the list.
    """
    prev = [m["content"] for m in history if m.get("role") == "user"]
    if prev and is_follow_up(question):
        return f"{prev[-1]}\n{question}"
    return question


def retrieve(repo: str, question: str, history: list[dict], top_k: int | None = None):
    qvec, tokens = vector_store.embed([retrieval_query(question, history)], "query")
    return vector_store.query(repo, qvec[0], top_k or config.TOP_K), tokens


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"### {c['path']} (lines {int(c['start_line'])}-{int(c['end_line'])})\n"
            f"```{c.get('language', '')}\n{c['code']}\n```"
        )
    return "\n\n".join(parts)


def citations(repo: str, commit_sha: str, chunks: list[dict]) -> list[dict]:
    """
    One entry per file with the line ranges of its retrieved chunks.
    Ranges are merged only when they overlap or touch; two chunks from the top
    and bottom of a file stay separate instead of becoming "L1-<end of file>".
    """
    by_path: dict[str, list[list[int]]] = {}
    for c in chunks:
        by_path.setdefault(c["path"], []).append([int(c["start_line"]), int(c["end_line"])])
    out = []
    for path, ranges in by_path.items():
        ranges.sort()
        merged = [ranges[0]]
        for s, e in ranges[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        base = f"https://github.com/{repo}/blob/{commit_sha}/{path}"
        out.append({
            "path": path,
            "ranges": [
                {"start_line": s, "end_line": e, "url": f"{base}#L{s}-L{e}"} for s, e in merged
            ],
        })
    return out


def stream_answer(repo: str, question: str, history: list[dict], chunks: list[dict]):
    """Yields answer text pieces as Gemini produces them."""
    yield from _get_chain().stream({
        "repo": repo,
        "context": format_context(chunks),
        "history": trim_history(history),
        "question": question,
    })
