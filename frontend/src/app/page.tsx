"use client";

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { Send, Terminal, Loader2, GitBranch, Link, ArrowRight, Code2 } from 'lucide-react';

export default function Home() {
  const [appState, setAppState] = useState<'landing' | 'ingesting' | 'chat'>('landing');
  const [repoUrl, setRepoUrl] = useState('');
  const [activeRepo, setActiveRepo] = useState('');
  
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string, citations?: string[]}[]>([]);
  const [loading, setLoading] = useState(false);

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setAppState('ingesting');
    
    try {
      // Clean up URL to pass just the owner/repo or full url
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await axios.post(`${apiUrl}/api/ingest`, { repo_url: repoUrl });
      setActiveRepo(response.data.repo_name);
      setAppState('chat');
      // Add welcome message
      setMessages([{
        role: 'assistant',
        content: `I've successfully analyzed **${response.data.repo_name}**! I've read all the code, documentation, and architecture. What would you like to know?`
      }]);
    } catch (error) {
      alert("Failed to ingest repository. Please check the URL and try again.");
      setAppState('landing');
    }
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setQuery('');
    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/api/chat`, { 
        message: userMessage,
        repo_url: activeRepo
      });
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: response.data.reply,
          citations: response.data.citations 
        }
      ]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error communicating with the backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  if (appState === 'landing' || appState === 'ingesting') {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
        {/* Navbar */}
        <nav className="w-full max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer">
            <Terminal className="w-7 h-7 text-blue-600" />
            <span className="text-2xl font-bold text-blue-600 tracking-tight">RepoTrace</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <span className="hover:text-blue-600 cursor-pointer transition-colors">About</span>
            <span className="hover:text-blue-600 cursor-pointer transition-colors">Resources</span>
            <span className="hover:text-blue-600 cursor-pointer transition-colors">Pricing</span>
            <span className="hover:text-blue-600 cursor-pointer transition-colors">Contact us</span>
          </div>
        </nav>

        {/* Main Content */}
        <div className="flex-1 flex flex-col items-center justify-center p-4">
          <div className="max-w-3xl w-full text-center space-y-8">
            <div className="flex justify-center">
              <div className="w-20 h-20 bg-blue-600 rounded-2xl flex items-center justify-center shadow-lg mb-4">
                <Code2 className="w-10 h-10 text-white" />
              </div>
            </div>
          
          <h1 className="text-5xl font-extrabold text-slate-900 tracking-tight">
            Chat with any Codebase.
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            RepoTrace uses AI to read, understand, and answer questions about any public GitHub repository. Get instant answers with exact file citations.
          </p>

          <div className="max-w-xl mx-auto mt-12">
            {appState === 'landing' ? (
              <form onSubmit={handleIngest} className="relative shadow-sm rounded-xl flex items-center bg-white border border-slate-300 overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all">
                <div className="pl-4">
                  <Link className="w-6 h-6 text-slate-400" />
                </div>
                <input
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
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
            ) : (
              <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-sm flex flex-col items-center">
                <Loader2 className="w-10 h-10 animate-spin text-blue-600 mb-4" />
                <h3 className="text-xl font-semibold text-slate-800 mb-2">Analyzing Repository...</h3>
                <p className="text-slate-500 text-center">
                  Downloading files, chunking code, and generating vector embeddings. 
                  <br/>This may take upto few minute due to API rate limits.
                </p>
              </div>
            )}
          </div>
          
          <div className="pt-12 text-sm text-slate-500 font-medium">
            Powered by LangChain, Pinecone, and Google Gemini
          </div>
        </div>
      </div>
      </div>
    );
  }

  // Chat State
  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 text-white p-6 flex flex-col border-r border-slate-800">
        <div className="flex items-center gap-3 mb-8 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => setAppState('landing')}>
          <Terminal className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-bold tracking-tight">RepoTrace</h1>
        </div>
        
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Active Repository</div>
        <div className="bg-slate-800 rounded-md p-3 flex items-center gap-2 border border-slate-700">
          <GitBranch className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium truncate">{activeRepo}</span>
        </div>
        
        <div className="mt-auto">
          <button 
            onClick={() => setAppState('landing')}
            className="w-full py-2 px-4 bg-slate-800 hover:bg-slate-700 rounded-md text-sm font-medium transition-colors border border-slate-700"
          >
            Analyze another repo
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col max-w-5xl mx-auto bg-white border-l border-r border-slate-200 shadow-sm relative w-full">
        
        {/* Header */}
        <div className="border-b border-slate-200 px-8 py-5 bg-white flex justify-between items-center z-10">
          <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Terminal className="w-5 h-5 text-blue-600" />
            Engineering Copilot
          </h2>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-slate-50">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] rounded-xl p-5 ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white shadow-sm' 
                  : 'bg-white border border-slate-200 text-slate-800 shadow-sm'
              }`}>
                <div className={`markdown text-sm leading-relaxed ${msg.role === 'user' ? 'text-white' : 'text-slate-700'}`}>
                  <ReactMarkdown>
                    {msg.content}
                  </ReactMarkdown>
                </div>
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-5 pt-4 border-t border-slate-100">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <GitBranch className="w-3 h-3" /> Sources cited
                    </p>
                    <ul className="flex flex-wrap gap-2">
                      {msg.citations.map((cite, i) => {
                        const filename = cite.split('/').pop() || cite;
                        return (
                          <li key={i} className="text-xs text-slate-600 bg-slate-100 hover:bg-slate-200 cursor-pointer px-2.5 py-1.5 rounded-md border border-slate-200 transition-colors flex items-center gap-1">
                            {filename}
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-5 flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                <span className="text-sm text-slate-600 font-medium">Generating response...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white border-t border-slate-200">
          <form onSubmit={handleChat} className="relative max-w-4xl mx-auto">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Ask a question about ${activeRepo}...`}
              className="w-full pl-5 pr-14 py-4 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-slate-900 shadow-sm transition-all text-sm font-medium placeholder:text-slate-400"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2.5 top-2.5 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors shadow-sm flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center mt-3">
            <span className="text-[11px] font-medium text-slate-400">AI can make mistakes. Always verify critical code citations.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
