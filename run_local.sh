#!/usr/bin/env bash
# Runs RepoTrace locally: backend on :8000, frontend on :5173.
# Backend output goes to backend/server.log. Stop everything with Ctrl+C.
set -e
cd "$(dirname "$0")"

PY=python3
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10+ is required (found $($PY --version))."; exit 1
fi

# Fresh virtualenv: the old backend/venv has both `pinecone` and `pinecone-client`, which clash.
if [ ! -d backend/.venv ]; then
  echo ">> creating backend/.venv"
  $PY -m venv backend/.venv
fi
echo ">> installing backend packages"
backend/.venv/bin/pip install -q --upgrade pip
backend/.venv/bin/pip install -q -r backend/requirements.txt
echo ">> installing optional tree-sitter grammars (failures are skipped)"
grep -v '^#' backend/requirements-ast-extra.txt | while read -r pkg; do
  [ -z "$pkg" ] && continue
  backend/.venv/bin/pip install -q "$pkg" >/dev/null 2>&1 && echo "   ok   $pkg" || echo "   skip $pkg"
done

echo ">> installing frontend packages"
(cd frontend && npm install --no-audit --no-fund)

echo ">> starting backend on http://localhost:8000 (log: backend/server.log)"
(cd backend && .venv/bin/uvicorn main:app --port 8000 --reload > server.log 2>&1) &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null' EXIT

echo ">> starting frontend on http://localhost:5173"
cd frontend
npm run dev
