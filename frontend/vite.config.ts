import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // In `npm run dev`, /api/* is forwarded to the local FastAPI server, so the
    // browser only talks to one origin (no CORS setup needed locally).
    proxy: { "/api": "http://localhost:8000" },
  },
});
