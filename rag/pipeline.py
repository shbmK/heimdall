"""The query-time RAG pipeline: retrieve, assemble prompt, generate.

``Retriever`` and ``RagPipeline`` are separate so retrieval can be evaluated
(and reused) without generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import RagConfig
from .llm import LLMProvider
from .store import BaseVectorStore, SearchHit

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


class Retriever:
    def __init__(self, config: RagConfig, client: LLMProvider, store: BaseVectorStore):
        self.config = config
        self.client = client
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[SearchHit]:
        query_vector = self.client.embed([query])[0]
        return self.store.search(query_vector, k=k or self.config.top_k)


class RagPipeline:
    def __init__(self, config: RagConfig, client: LLMProvider, store: BaseVectorStore):
        self.config = config
        self.client = client
        self.retriever = Retriever(config, client, store)

    def _build_prompt(self, question: str, hits: list[SearchHit]) -> str:
        parts = []
        budget = self.config.max_context_chars
        for i, hit in enumerate(hits, start=1):
            passage = hit.chunk.text
            if len(passage) > budget:
                passage = passage[:budget]
            budget -= len(passage)
            source = hit.chunk.metadata.get("title", hit.chunk.doc_id)
            parts.append(f"[{i}] (from: {source})\n{passage}")
            if budget <= 0:
                break
        context = "\n\n---\n\n".join(parts)
        return f"Context passages:\n\n{context}\n\nQuestion: {question}"

    def answer(self, question: str, k: int | None = None) -> RagAnswer:
        start = time.perf_counter()
        hits = self.retriever.retrieve(question, k=k)
        retrieval_seconds = time.perf_counter() - start

        prompt = self._build_prompt(question, hits)
        text = self.client.generate(prompt, system=SYSTEM_PROMPT)
        total_seconds = time.perf_counter() - start

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
        )
