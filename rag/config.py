"""Central configuration for the RAG pipeline.

Every value can be overridden with an environment variable prefixed with
``RAG_`` (e.g. ``RAG_CHAT_MODEL=llama3.1:8b rag ask "..."``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"RAG_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"RAG_{name}")
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"RAG_{name}")
    return float(raw) if raw else default


@dataclass
class RagConfig:
    # LLM provider: 'ollama' (default, local) or 'openai' (OpenAI-compatible API)
    llm_provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "ollama"))
    embed_model: str = field(default_factory=lambda: _env_str("EMBED_MODEL", "nomic-embed-text"))
    chat_model: str = field(default_factory=lambda: _env_str("CHAT_MODEL", "llama3.2:3b"))
    # Judge defaults to the chat model; override for a stronger judge if available.
    judge_model: str = field(default_factory=lambda: _env_str("JUDGE_MODEL", ""))
    request_timeout: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT", 300.0))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))

    # Ollama provider
    ollama_host: str = field(default_factory=lambda: _env_str("OLLAMA_HOST", "http://127.0.0.1:11434"))

    # OpenAI-compatible provider
    openai_base_url: str = field(default_factory=lambda: _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))

    # Chunking
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1800))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 250))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))

    # Vector store: 'local' (numpy, default) or 'qdrant' (external vector DB)
    vector_backend: str = field(default_factory=lambda: _env_str("VECTOR_BACKEND", "local"))
    qdrant_url: str = field(default_factory=lambda: _env_str("QDRANT_URL", "http://127.0.0.1:6333"))
    qdrant_collection: str = field(default_factory=lambda: _env_str("QDRANT_COLLECTION", "rag_base"))
    qdrant_upsert_batch_size: int = field(default_factory=lambda: _env_int("QDRANT_UPSERT_BATCH_SIZE", 256))

    # Generation
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.0))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 12000))

    # Paths (relative to the current working directory by default)
    corpus_dir: Path = field(default_factory=lambda: Path(_env_str("CORPUS_DIR", "data/corpus")))
    index_dir: Path = field(default_factory=lambda: Path(_env_str("INDEX_DIR", "data/index")))
    eval_path: Path = field(default_factory=lambda: Path(_env_str("EVAL_PATH", "data/eval/eval_set.json")))
    eval_results_path: Path = field(default_factory=lambda: Path(_env_str("EVAL_RESULTS_PATH", "eval_results.json")))

    # Scraping
    scrape_delay_seconds: float = field(default_factory=lambda: _env_float("SCRAPE_DELAY_SECONDS", 0.5))
    max_doc_chars: int = field(default_factory=lambda: _env_int("MAX_DOC_CHARS", 60000))

    def effective_judge_model(self) -> str:
        return self.judge_model or self.chat_model


def load_config() -> RagConfig:
    return RagConfig()
