"""Pluggable LLM providers (embeddings + chat generation).

``LLMProvider`` is the interface the rest of the pipeline depends on; nothing
outside this module knows which provider is active. Select one with
``RAG_LLM_PROVIDER`` (default ``ollama``).

Providers:
- ``ollama`` (default): local models via the Ollama HTTP API. No API key.
- ``openai``: any OpenAI-compatible REST endpoint (OpenAI itself, or local
  servers like vLLM / LM Studio / llama.cpp that expose ``/v1``). Reads the
  API key from ``OPENAI_API_KEY`` (or ``RAG_OPENAI_API_KEY``).

Both talk HTTP via ``requests`` with bounded retries and exponential backoff;
neither requires a vendor SDK. Adding a provider means subclassing
``LLMProvider`` and registering it in ``create_llm``.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import numpy as np
import requests

from .config import RagConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM backend fails after all retries."""


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int,
    timeout: float,
    label: str,
    **kwargs,
) -> dict:
    """Issue an HTTP request with exponential backoff, returning parsed JSON."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                backoff = 2**attempt
                logger.warning(
                    "%s attempt=%d/%d backoff_s=%d error=%s",
                    label,
                    attempt + 1,
                    max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
    logger.error("%s attempts=%d error=%s", label, max_retries, last_error)
    raise LLMError(f"{label} failed after {max_retries} attempts: {last_error}")


class LLMProvider(ABC):
    """Contract between the pipeline and any embedding + chat backend."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts; returns an (n, dim) float32 array."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a chat completion for a single prompt."""

    @abstractmethod
    def available_models(self) -> list[str]:
        """List model identifiers the backend currently exposes."""

    @abstractmethod
    def check_ready(self, models: list[str]) -> None:
        """Raise LLMError with actionable guidance if the backend or the
        required models aren't usable."""

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable status line for `rag info`."""


# --------------------------------------------------------------------- ollama


class OllamaProvider(LLMProvider):
    def __init__(self, config: RagConfig):
        self.config = config
        self.base_url = config.ollama_host.rstrip("/")
        self.session = requests.Session()

    def _post(self, path: str, payload: dict) -> dict:
        return _request_with_retries(
            self.session,
            "POST",
            f"{self.base_url}{path}",
            max_retries=self.config.max_retries,
            timeout=self.config.request_timeout,
            label=f"Ollama request to {path}",
            json=payload,
        )

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
        missing = [m for m in models if m not in available and f"{m}:latest" not in available]
        if missing:
            logger.error("ollama missing models=%s available=%d", ", ".join(missing), len(available))
            raise LLMError(
                f"Missing Ollama models: {', '.join(missing)}. "
                f"Run: {'; '.join(f'ollama pull {m}' for m in missing)}"
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = self.config.embed_batch_size
        logger.debug("ollama embed texts=%d batch_size=%d model=%s", len(texts), batch_size, self.config.embed_model)
        start = time.perf_counter()
        for start_idx in range(0, len(texts), batch_size):
            batch = texts[start_idx : start_idx + batch_size]
            data = self._post("/api/embed", {"model": self.config.embed_model, "input": batch})
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise LLMError(f"Embedding API returned {len(embeddings or [])} vectors for {len(batch)} inputs")
            vectors.extend(embeddings)
        elapsed = time.perf_counter() - start
        result = np.asarray(vectors, dtype=np.float32)
        logger.info("ollama embed texts=%d dim=%d elapsed_s=%.2f", len(texts), result.shape[1] if result.size else 0, elapsed)
        return result

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
        model_name = model or self.config.chat_model
        logger.debug("ollama generate model=%s prompt_chars=%d", model_name, len(prompt))
        start = time.perf_counter()
        data = self._post(
            "/api/chat",
            {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config.temperature if temperature is None else temperature},
            },
        )
        try:
            text = data["message"]["content"].strip()
            logger.info("ollama generate model=%s response_chars=%d elapsed_s=%.2f", model_name, len(text), time.perf_counter() - start)
            return text
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {data}") from exc

    def describe(self) -> str:
        return f"ollama @ {self.base_url}"


# --------------------------------------------------------------------- openai


class OpenAIProvider(LLMProvider):
    """Works against any OpenAI-compatible ``/v1`` endpoint."""

    def __init__(self, config: RagConfig):
        self.config = config
        self.base_url = config.openai_base_url.rstrip("/")
        self.api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not self.api_key and not self._is_local:
            raise LLMError(
                "The openai provider needs an API key. Set OPENAI_API_KEY "
                "(or RAG_OPENAI_API_KEY), or point RAG_OPENAI_BASE_URL at a local server."
            )
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _post(self, path: str, payload: dict) -> dict:
        return _request_with_retries(
            self.session,
            "POST",
            f"{self.base_url}{path}",
            max_retries=self.config.max_retries,
            timeout=self.config.request_timeout,
            label=f"OpenAI request to {path}",
            json=payload,
        )

    def available_models(self) -> list[str]:
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return [m["id"] for m in response.json().get("data", [])]
        except requests.RequestException as exc:
            raise LLMError(f"Cannot reach the OpenAI-compatible endpoint at {self.base_url}: {exc}") from exc

    def check_ready(self, models: list[str]) -> None:
        # Many compatible servers don't implement /models; treat listing as a
        # soft check and only fail on a hard connection error.
        try:
            available = set(self.available_models())
        except LLMError:
            return
        if available:
            missing = [m for m in models if m not in available]
            if missing:
                logger.error("openai missing models=%s", ", ".join(missing))
                raise LLMError(
                    f"Models not available on {self.base_url}: {', '.join(missing)}. "
                    f"Available: {', '.join(sorted(available)[:10])}..."
                )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = self.config.embed_batch_size
        logger.debug("openai embed texts=%d batch_size=%d model=%s", len(texts), batch_size, self.config.embed_model)
        start = time.perf_counter()
        for start_idx in range(0, len(texts), batch_size):
            batch = texts[start_idx : start_idx + batch_size]
            data = self._post("/embeddings", {"model": self.config.embed_model, "input": batch})
            items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            if len(items) != len(batch):
                raise LLMError(f"Embedding API returned {len(items)} vectors for {len(batch)} inputs")
            vectors.extend(item["embedding"] for item in items)
        elapsed = time.perf_counter() - start
        result = np.asarray(vectors, dtype=np.float32)
        logger.info("openai embed texts=%d dim=%d elapsed_s=%.2f", len(texts), result.shape[1] if result.size else 0, elapsed)
        return result

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
        model_name = model or self.config.chat_model
        logger.debug("openai generate model=%s prompt_chars=%d", model_name, len(prompt))
        start = time.perf_counter()
        data = self._post(
            "/chat/completions",
            {
                "model": model_name,
                "messages": messages,
                "temperature": self.config.temperature if temperature is None else temperature,
            },
        )
        try:
            text = data["choices"][0]["message"]["content"].strip()
            logger.info("openai generate model=%s response_chars=%d elapsed_s=%.2f", model_name, len(text), time.perf_counter() - start)
            return text
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {data}") from exc

    def describe(self) -> str:
        auth = "keyed" if self.api_key else "no key"
        return f"openai-compatible @ {self.base_url} ({auth})"


# ------------------------------------------------------------------- factory

_PROVIDERS = ("ollama", "openai")


def create_llm(config: RagConfig) -> LLMProvider:
    provider = config.llm_provider
    logger.info("llm provider=%s", provider)
    if provider == "ollama":
        return OllamaProvider(config)
    if provider == "openai":
        return OpenAIProvider(config)
    raise LLMError(f"Unknown LLM provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}")
