"""Thin, robust client for the Ollama HTTP API.

Uses plain ``requests`` (no SDK dependency) with bounded retries and
exponential backoff. All pipeline components talk to Ollama exclusively
through this module, so swapping the backend means changing one class.
"""

from __future__ import annotations

import time

import numpy as np
import requests

from .config import RagConfig


class LLMError(RuntimeError):
    """Raised when the LLM backend fails after all retries."""


class OllamaClient:
    def __init__(self, config: RagConfig):
        self.config = config
        self.base_url = config.ollama_host.rstrip("/")
        self.session = requests.Session()

    # ------------------------------------------------------------------ http
    def _post(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    timeout=self.config.request_timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        raise LLMError(f"Ollama request to {path} failed after {self.config.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------ diagnostics
    def available_models(self) -> list[str]:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except requests.RequestException as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running? ({exc})"
            ) from exc

    def check_ready(self, models: list[str]) -> None:
        available = self.available_models()
        missing = [
            m for m in models
            if m not in available and f"{m}:latest" not in available
        ]
        if missing:
            raise LLMError(
                f"Missing Ollama models: {', '.join(missing)}. "
                f"Run: {'; '.join(f'ollama pull {m}' for m in missing)}"
            )

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, batching requests. Returns (n, dim) float32."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = self.config.embed_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = self._post("/api/embed", {"model": self.config.embed_model, "input": batch})
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise LLMError(f"Embedding API returned {len(embeddings or [])} vectors for {len(batch)} inputs")
            vectors.extend(embeddings)
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------- generation
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self._post(
            "/api/chat",
            {
                "model": model or self.config.chat_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature if temperature is None else temperature,
                },
            },
        )
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {data}") from exc
