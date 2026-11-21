import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "highlight.js/styles/github-dark.css";
import "./index.css";
import Landing from "./pages/Landing";
import RepoChat from "./pages/RepoChat";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        {/* A URL per repo: refresh keeps you in the chat and links can be shared. */}
        <Route path="/repo/:owner/:name" element={<RepoChat />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
