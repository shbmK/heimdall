"""The query-time RAG pipeline: retrieve, assemble prompt, generate.

``Retriever`` and ``RagPipeline`` are separate so retrieval can be evaluated
(and reused) without generation.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .config import RagConfig
from .fallback import FallbackResult, fetch_and_ingest
from .llm import LLMProvider
from .logging_config import get_logger
from .store import BaseVectorStore, SearchHit

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a comics knowledge assistant. Answer the user's question using ONLY the provided context passages about Marvel and DC characters.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- Cite the passages you used by their number, e.g. [1] or [2][3].
- Be concise: a few sentences, directly answering the question.
- If the context does not contain the information needed, reply exactly: "I don't know — that information is not in my knowledge base." Do not guess."""


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[SearchHit]
    retrieval_seconds: float
    total_seconds: float
    sources: list[dict] = field(default_factory=list)
    fallback_used: bool = False
    fallback_doc: str = ""


class _EmbedCache:
    """Simple LRU cache for query embedding vectors."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, key: tuple[str, str]) -> np.ndarray | None:
        if self.maxsize <= 0 or key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: tuple[str, str], value: np.ndarray) -> None:
        if self.maxsize <= 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


class Retriever:
    def __init__(self, config: RagConfig, client: LLMProvider, store: BaseVectorStore):
        self.config = config
        self.client = client
        self.store = store
        self._embed_cache = _EmbedCache(config.embed_cache_size)

    def embed_query(self, query: str) -> np.ndarray:
        key = (self.config.embed_model, query.strip().lower())
        cached = self._embed_cache.get(key)
        if cached is not None:
            logger.debug("embed cache hit model=%s", self.config.embed_model)
            return cached
        logger.debug("embed cache miss model=%s chars=%d", self.config.embed_model, len(query))
        vector = self.client.embed([query])[0]
        self._embed_cache.put(key, vector)
        return vector

    def retrieve(self, query: str, k: int | None = None) -> list[SearchHit]:
        k = k or self.config.top_k
        query_vector = self.embed_query(query)
        return self.store.search(query_vector, k=k, query_text=query)

    def retrieve_with_vector(
        self,
        query_vector: np.ndarray,
        k: int | None = None,
        query_text: str | None = None,
    ) -> list[SearchHit]:
        return self.store.search(
            query_vector, k=k or self.config.top_k, query_text=query_text
        )


class RagPipeline:
    def __init__(self, config: RagConfig, client: LLMProvider, store: BaseVectorStore):
        self.config = config
        self.client = client
        self.store = store
        self.retriever = Retriever(config, client, store)

    def _build_prompt(self, question: str, hits: list[SearchHit]) -> str:
        parts = []
        budget = self.config.max_context_chars
        used_hits = 0
        for i, hit in enumerate(hits, start=1):
            passage = hit.chunk.text
            if len(passage) > budget:
                passage = passage[:budget]
            budget -= len(passage)
            source = hit.chunk.metadata.get("title", hit.chunk.doc_id)
            parts.append(f"[{i}] (from: {source})\n{passage}")
            used_hits = i
            if budget <= 0:
                logger.debug("context budget exhausted hits=%d/%d", used_hits, len(hits))
                break
        context = "\n\n---\n\n".join(parts)
        logger.debug("prompt hits=%d chars=%d", used_hits, len(context))
        return f"Context passages:\n\n{context}\n\nQuestion: {question}"

    @staticmethod
    def _needs_fallback(hits: list[SearchHit], min_score: float) -> bool:
        if not hits:
            return True
        return max(h.score for h in hits) < min_score

    def _try_fallback(self, question: str) -> FallbackResult | None:
        if not self.config.fallback_enabled:
            logger.debug("fallback disabled")
            return None
        return fetch_and_ingest(self.config, self.client, self.store, question)

    def answer(self, question: str, k: int | None = None) -> RagAnswer:
        k = k or self.config.top_k
        logger.info("query k=%d %s", k, question[:120])
        start = time.perf_counter()
        query_vector = self.retriever.embed_query(question)
        hits = self.retriever.retrieve_with_vector(query_vector, k=k, query_text=question)

        max_score = max((h.score for h in hits), default=0.0)
        logger.info(
            "retrieve hits=%d max_score=%.4f threshold=%.4f docs=%s",
            len(hits),
            max_score,
            self.config.fallback_min_score,
            [h.chunk.doc_id for h in hits[:3]],
        )

        fallback_used = False
        fallback_doc = ""
        if self._needs_fallback(hits, self.config.fallback_min_score):
            logger.warning(
                "low score max=%.4f threshold=%.4f",
                max_score,
                self.config.fallback_min_score,
            )
            result = self._try_fallback(question)
            if result is not None and result.ok:
                fallback_used = True
                fallback_doc = result.doc_id
                logger.info("fallback ok doc=%s chunks=%d", result.doc_id, result.chunks_added)
                hits = self.retriever.retrieve_with_vector(
                    query_vector, k=k, query_text=question
                )
                max_score = max((h.score for h in hits), default=0.0)
                logger.info("retrieve post-fallback hits=%d max_score=%.4f", len(hits), max_score)
            elif result is not None:
                logger.warning(
                    "fallback failed skipped_existing=%s error=%s",
                    result.skipped_existing,
                    result.error,
                )

        retrieval_seconds = time.perf_counter() - start

        prompt = self._build_prompt(question, hits)
        text = self.client.generate(prompt, system=SYSTEM_PROMPT)
        total_seconds = time.perf_counter() - start

        logger.info(
            "query done retrieval_ms=%.0f total_s=%.2f fallback=%s answer_chars=%d",
            retrieval_seconds * 1000,
            total_seconds,
            fallback_used,
            len(text),
        )

        sources = [
            {
                "rank": i + 1,
                "doc": h.chunk.doc_id,
                "title": h.chunk.metadata.get("title", ""),
                "section": h.chunk.section,
                "score": round(h.score, 4),
                "url": h.chunk.metadata.get("source", ""),
            }
            for i, h in enumerate(hits)
        ]
        return RagAnswer(
            question=question,
            answer=text,
            hits=hits,
            retrieval_seconds=retrieval_seconds,
            total_seconds=total_seconds,
            sources=sources,
            fallback_used=fallback_used,
            fallback_doc=fallback_doc,
        )
