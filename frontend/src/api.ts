// Dev: same origin, forwarded to localhost:8000 by the Vite proxy (vite.config.ts).
// Production: the Render backend URL set in Vercel as VITE_API_URL.
const API = import.meta.env.DEV ? "" : (import.meta.env.VITE_API_URL ?? "");

export type JobStatus = {
  status: "queued" | "running" | "ready" | "failed";
  repo: string;
  stage?: string;
  files?: number;
  chunks_total?: number;
  chunks_done?: number;
  est_tokens?: number;
  error?: string | null;
  commit_sha?: string;
};

export type LineRange = { start_line: number; end_line: number; url: string };

export type Citation = { path: string; ranges: LineRange[] };

export type ChatTurn = { role: "user" | "assistant"; content: string };

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `Request failed (${res.status})`;
}

/** Starts (or reuses) ingestion. Resolves with "ready" or a running job. */
export async function startIngest(repoUrl: string): Promise<JobStatus> {
  const res = await fetch(`${API}/api/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repoUrl }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function getStatus(repo: string): Promise<JobStatus | null> {
  const res = await fetch(`${API}/api/repos/${repo}/status`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

type StreamHandlers = {
  onCitations: (c: Citation[]) => void;
  onToken: (text: string) => void;
};

/** Reads the backend's newline-delimited JSON stream. */
export async function streamChat(
  repo: string,
  message: string,
  history: ChatTurn[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, message, history }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorDetail(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // last piece may be an incomplete line
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "citations") handlers.onCitations(ev.citations);
      else if (ev.type === "token") handlers.onToken(ev.text);
      else if (ev.type === "error") throw new Error(ev.detail);
    }
  }
}
