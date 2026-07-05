"""Build the vector index from the corpus directory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_document
from .config import RagConfig
from .llm import OllamaClient
from .store import create_store


@dataclass
class IngestStats:
    documents: int
    chunks: int
    backend: str
    location: str


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse the simple ``key: value`` front matter our scraper writes."""
    meta: dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            text = text[end + 5 :]
    return meta, text.lstrip()


def load_corpus(corpus_dir: Path) -> list[tuple[str, str, str, dict]]:
    """Return (doc_id, title, body, metadata) for each document in the corpus."""
    docs = []
    for path in sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        title = meta.get("title")
        if not title:
            m = re.match(r"^#\s+(.+)$", body, flags=re.MULTILINE)
            title = m.group(1).strip() if m else path.stem
        docs.append((path.name, title, body, meta))
    return docs


def build_index(config: RagConfig, client: OllamaClient, progress=None) -> IngestStats:
    docs = load_corpus(config.corpus_dir)
    if not docs:
        raise FileNotFoundError(
            f"No .md/.txt documents in {config.corpus_dir}. Run `rag scrape` first (or drop your own files there)."
        )

    all_chunks: list[Chunk] = []
    for doc_id, title, body, meta in docs:
        chunks = chunk_document(
            doc_id=doc_id,
            title=title,
            text=body,
            max_chars=config.chunk_chars,
            overlap_chars=config.chunk_overlap_chars,
            metadata={"title": title, "universe": meta.get("universe", ""), "source": meta.get("source", "")},
        )
        all_chunks.extend(chunks)
        if progress:
            progress(doc_id, len(chunks))

    store = create_store(config)
    embeddings = client.embed([c.text for c in all_chunks])
    store.add(all_chunks, embeddings)
    store.persist()
    location = config.qdrant_url if config.vector_backend == "qdrant" else str(config.index_dir)
    return IngestStats(
        documents=len(docs), chunks=len(all_chunks), backend=config.vector_backend, location=location
    )
