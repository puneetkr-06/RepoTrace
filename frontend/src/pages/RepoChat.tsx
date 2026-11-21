import { useEffect, useRef, useState, type FormEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { AlertTriangle, FileCode, GitBranch, Loader2, Send, Terminal } from "lucide-react";
import { getStatus, startIngest, streamChat, type ChatTurn, type Citation, type JobStatus } from "../api";

// The status endpoint only reads an in-memory object on the server, so a poll
// is cheap. Ingestion takes minutes (embedding alone is limited to 250K
// tokens/min), so checking every 3 s keeps the progress bar live at <= 20
// requests per minute.
const POLL_MS = 3000;

type Message = ChatTurn & { citations?: Citation[]; error?: boolean };

const STAGE_LABEL: Record<string, string> = {
  queued: "Waiting to start",
  downloading: "Downloading repository",
  chunking: "Splitting code into chunks",
  embedding: "Generating embeddings",
  finalizing: "Finishing up",
};

function storageKey(repo: string) {
  return `repotrace:chat:${repo}`;
}

function loadMessages(repo: string): Message[] {
  try {
    const raw = sessionStorage.getItem(storageKey(repo));
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}

function saveMessages(repo: string, msgs: Message[]) {
  try {
    sessionStorage.setItem(storageKey(repo), JSON.stringify(msgs));
  } catch {
    /* storage full or unavailable: chat still works, just not after refresh */
  }
}

export default function RepoChat() {
  const { owner, name } = useParams();
  const navigate = useNavigate();
  const requested = `${owner}/${name}`;

  const [repo, setRepo] = useState(requested);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [phase, setPhase] = useState<"checking" | "ingesting" | "ready" | "failed">("checking");
  const [failure, setFailure] = useState("");

  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // 1. Start or reuse ingestion, then poll until ready.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const becomeReady = (canonical: string) => {
      setRepo(canonical);
      setMessages(loadMessages(canonical));
      setPhase("ready");
      if (canonical !== requested) navigate(`/repo/${canonical}`, { replace: true });
    };

    const poll = async (canonical: string) => {
      try {
        const s = await getStatus(canonical);
        if (cancelled) return;
        if (!s) throw new Error("Ingestion stopped unexpectedly. Please try again.");
        setJob(s);
        if (s.status === "ready") becomeReady(canonical);
        else if (s.status === "failed") {
          setFailure(s.error ?? "Ingestion failed.");
          setPhase("failed");
        } else timer = setTimeout(() => poll(canonical), POLL_MS);
      } catch (e) {
        if (!cancelled) {
          setFailure((e as Error).message);
          setPhase("failed");
        }
      }
    };

    (async () => {
      setPhase("checking");
      try {
        const s = await startIngest(requested);
        if (cancelled) return;
        if (s.status === "ready") becomeReady(s.repo);
        else {
          setJob(s);
          setRepo(s.repo);
          setPhase("ingesting");
          timer = setTimeout(() => poll(s.repo), POLL_MS);
        }
      } catch (e) {
        if (!cancelled) {
          setFailure((e as Error).message);
          setPhase("failed");
        }
      }
    })();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [requested, navigate]);

  useEffect(() => {
    if (phase === "ready") saveMessages(repo, messages);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase, repo]);

  // 2. Chat with a streamed answer.
  const send = async (e: FormEvent) => {
    e.preventDefault();
    const text = query.trim();
    if (!text || streaming) return;
    // The server keeps only the most recent ~4,000 characters of history
    // anyway, so sending the last 20 messages is always enough.
    const history: ChatTurn[] = messages
      .filter((m) => !m.error && m.content)
      .slice(-20)
      .map(({ role, content }) => ({ role, content }));
    setQuery("");
    setStreaming(true);
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);

    const updateLast = (fn: (m: Message) => Message) =>
      setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);

    try {
      await streamChat(repo, text, history, {
        onCitations: (citations) => updateLast((m) => ({ ...m, citations })),
        onToken: (t) => updateLast((m) => ({ ...m, content: m.content + t })),
      });
    } catch (err) {
      updateLast((m) => ({
        ...m,
        error: true,
        content: m.content || `Sorry, something went wrong: ${(err as Error).message}`,
      }));
    } finally {
      setStreaming(false);
    }
  };

  if (phase !== "ready") {
    const pct =
      job?.chunks_total && job.chunks_total > 0
        ? Math.round(((job.chunks_done ?? 0) / job.chunks_total) * 100)
        : null;
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
        <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-sm w-full max-w-lg flex flex-col items-center text-center">
          {phase === "failed" ? (
            <>
              <AlertTriangle className="w-10 h-10 text-red-500 mb-4" />
              <h3 className="text-xl font-semibold text-slate-800 mb-2">Could not analyze {repo}</h3>
              <p className="text-slate-600 mb-6">{failure}</p>
              <button
                onClick={() => navigate("/")}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium"
              >
                Try another repository
              </button>
            </>
          ) : (
            <>
              <Loader2 className="w-10 h-10 animate-spin text-blue-600 mb-4" />
              <h3 className="text-xl font-semibold text-slate-800 mb-1">Analyzing {repo}</h3>
              <p className="text-slate-500 mb-5">
                {phase === "checking" ? "Checking repository..." : STAGE_LABEL[job?.stage ?? "queued"] ?? job?.stage}
              </p>
              {pct !== null && (
                <div className="w-full">
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-600 transition-all" style={{ width: `${pct}%` }} />
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    {job?.chunks_done} / {job?.chunks_total} chunks from {job?.files} files ({pct}%)
                  </p>
                </div>
              )}
              <p className="text-xs text-slate-400 mt-5">You can leave this page open; it updates by itself.</p>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900">
      <aside className="w-64 bg-slate-900 text-white p-6 hidden md:flex flex-col border-r border-slate-800">
        <button
          className="flex items-center gap-3 mb-8 hover:opacity-80 transition-opacity"
          onClick={() => navigate("/")}
        >
          <Terminal className="w-6 h-6 text-blue-500" />
          <span className="text-xl font-bold tracking-tight">RepoTrace</span>
        </button>
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Active Repository</div>
        <a
          href={`https://github.com/${repo}`}
          target="_blank"
          rel="noreferrer"
          className="bg-slate-800 rounded-md p-3 flex items-center gap-2 border border-slate-700 hover:border-slate-500"
        >
          <GitBranch className="w-4 h-4 text-slate-400 shrink-0" />
          <span className="text-sm font-medium truncate">{repo}</span>
        </a>
        <div className="mt-auto space-y-2">
          <button
            onClick={() => setMessages([])}
            className="w-full py-2 px-4 bg-slate-800 hover:bg-slate-700 rounded-md text-sm font-medium border border-slate-700"
          >
            Clear chat
          </button>
          <button
            onClick={() => navigate("/")}
            className="w-full py-2 px-4 bg-slate-800 hover:bg-slate-700 rounded-md text-sm font-medium border border-slate-700"
          >
            Analyze another repo
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col max-w-5xl mx-auto bg-white border-x border-slate-200 w-full">
        <header className="border-b border-slate-200 px-8 py-5 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-blue-600" />
          <h2 className="text-xl font-bold tracking-tight">Engineering Copilot</h2>
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-slate-50">
          {messages.length === 0 && (
            <div className="text-center text-slate-500 mt-16">
              <p className="font-medium text-slate-700">{repo} is ready.</p>
              <p className="text-sm mt-1">Ask how something works, where it lives, or why it is built that way.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-xl p-5 shadow-sm ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : m.error
                      ? "bg-red-50 border border-red-200 text-red-800"
                      : "bg-white border border-slate-200 text-slate-700"
                }`}
              >
                {m.role === "assistant" && !m.content && streaming && i === messages.length - 1 ? (
                  <span className="flex items-center gap-2 text-sm text-slate-500">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-600" /> Searching the code...
                  </span>
                ) : (
                  <div className="markdown text-sm leading-relaxed">
                    <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{m.content}</ReactMarkdown>
                  </div>
                )}
                {m.citations && m.citations.length > 0 && (
                  <div className="mt-5 pt-4 border-t border-slate-100">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <FileCode className="w-3 h-3" /> Sources
                    </p>
                    <ul className="flex flex-wrap gap-2">
                      {m.citations.flatMap((c) =>
                        // "?? []": messages saved before ranges existed have no `ranges` field
                        (c.ranges ?? []).map((r) => (
                          <li key={`${c.path}:${r.start_line}`}>
                            <a
                              href={r.url}
                              target="_blank"
                              rel="noreferrer"
                              title={c.path}
                              className="text-xs text-slate-600 bg-slate-100 hover:bg-slate-200 px-2.5 py-1.5 rounded-md border border-slate-200 inline-block"
                            >
                              {c.path.split("/").pop()}{" "}
                              <span className="text-slate-400">
                                L{r.start_line}-{r.end_line}
                              </span>
                            </a>
                          </li>
                        )),
                      )}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="p-6 bg-white border-t border-slate-200">
          <form onSubmit={send} className="relative max-w-4xl mx-auto">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              maxLength={2000}
              placeholder={`Ask a question about ${repo}...`}
              className="w-full pl-5 pr-14 py-4 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm font-medium placeholder:text-slate-400"
              disabled={streaming}
            />
            <button
              type="submit"
              disabled={streaming || !query.trim()}
              className="absolute right-2.5 top-2.5 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              aria-label="Send"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <p className="text-center mt-3 text-[11px] font-medium text-slate-400">
            AI can make mistakes. Check the cited lines before relying on an answer.
          </p>
        </div>
      </main>
    </div>
  );
}
