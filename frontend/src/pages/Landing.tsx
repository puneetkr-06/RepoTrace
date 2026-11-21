import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Code2, Link, Terminal } from "lucide-react";

const REPO_RE =
  /^(?:https?:\/\/)?(?:www\.)?(?:github\.com\/)?([A-Za-z0-9][A-Za-z0-9-]{0,38})\/([A-Za-z0-9._-]+?)(?:\.git)?(?:\/.*)?$/;

export function parseRepo(input: string): string | null {
  const m = input.trim().replace(/\/+$/, "").match(REPO_RE);
  return m ? `${m[1]}/${m[2]}` : null;
}

export default function Landing() {
  const [repoUrl, setRepoUrl] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const repo = parseRepo(repoUrl);
    if (!repo) {
      setError("Enter a GitHub repository as owner/repo or a github.com URL.");
      return;
    }
    navigate(`/repo/${repo}`);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <nav className="w-full max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-7 h-7 text-blue-600" />
          <span className="text-2xl font-bold text-blue-600 tracking-tight">RepoTrace</span>
        </div>
        <a
          href="https://github.com/puneetkr-06/RepoTrace"
          target="_blank"
          rel="noreferrer"
          className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors"
        >
          Source on GitHub
        </a>
      </nav>

      <div className="flex-1 flex flex-col items-center justify-center p-4">
        <div className="max-w-3xl w-full text-center space-y-8">
          <div className="flex justify-center">
            <div className="w-20 h-20 bg-blue-600 rounded-2xl flex items-center justify-center shadow-lg mb-4">
              <Code2 className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-extrabold text-slate-900 tracking-tight">Chat with any Codebase.</h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            RepoTrace reads any public GitHub repository and answers questions about it, citing the
            exact files and lines it used.
          </p>

          <div className="max-w-xl mx-auto mt-12">
            <form
              onSubmit={submit}
              className="relative shadow-sm rounded-xl flex items-center bg-white border border-slate-300 overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all"
            >
              <div className="pl-4">
                <Link className="w-6 h-6 text-slate-400" />
              </div>
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => {
                  setRepoUrl(e.target.value);
                  setError("");
                }}
                placeholder="Paste a GitHub repository URL (e.g. facebook/react)"
                className="w-full py-5 px-4 text-slate-900 focus:outline-none bg-transparent placeholder:text-slate-400 font-medium"
                required
              />
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-5 font-semibold flex items-center gap-2 transition-colors border-l border-blue-700"
              >
                Analyze <ArrowRight className="w-5 h-5" />
              </button>
            </form>
            {error && <p className="mt-3 text-sm text-red-600 text-left">{error}</p>}
          </div>

          <div className="pt-12 text-sm text-slate-500 font-medium">
            Built with React, FastAPI, LangChain, Pinecone and Google Gemini
          </div>
        </div>
      </div>
    </div>
  );
}
