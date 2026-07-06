#!/usr/bin/env bash
# wait for ollama + qdrant, build the index once, then hand off to CMD
set -euo pipefail

python - <<'PY'
import os, sys, time
import requests

targets = [
    (os.environ.get("RAG_OLLAMA_HOST", "http://ollama:11434") + "/api/version", "ollama"),
    (os.environ.get("RAG_QDRANT_URL", "http://qdrant:6333") + "/readyz", "qdrant"),
]

def wait(url, name, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                print(f"[entrypoint] {name} is ready", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    sys.exit(f"[entrypoint] timed out waiting for {name} at {url}")

for url, name in targets:
    print(f"[entrypoint] waiting for {name} at {url} ...", flush=True)
    wait(url, name)
PY

if [ "${RAG_AUTO_INGEST:-1}" = "1" ]; then
    echo "[entrypoint] building index (rag ingest) ..."
    # don't kill the container if ingest fails; leave it up so it can be retried
    rag ingest || echo "[entrypoint] WARNING: ingest failed; run 'docker compose exec rag rag ingest' to retry"
fi

echo "[entrypoint] ready. Try: docker compose exec rag rag ask \"Where is Themyscira?\""
exec "$@"
