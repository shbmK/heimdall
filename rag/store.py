"""Pluggable vector store backends.

``BaseVectorStore`` defines the interface the rest of the pipeline uses;
nothing outside this module knows which backend is active.

Backends:
- ``local`` (default): numpy cosine-similarity index persisted to disk.
  Zero external services, ideal for corpora up to a few hundred thousand
  chunks. Supports hybrid dense+BM25 scoring and per-document diversity.
- ``qdrant``: a real vector database, for larger corpora or shared access.
  Requires ``pip install rag-base[qdrant]`` and a running Qdrant server
  (``docker run -p 6333:6333 qdrant/qdrant``). Select with
  ``RAG_VECTOR_BACKEND=qdrant``. Hybrid BM25 is applied over the dense
  candidate set (not the full collection).

Use ``create_store(config)`` when (re)building an index and
``open_store(config)`` when querying an existing one.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .config import RagConfig
from .logging_config import get_logger
from .sparse import BM25

logger = get_logger(__name__)

_VECTORS_FILE = "vectors.npz"
_CHUNKS_FILE = "chunks.json"


class StoreError(RuntimeError):
    pass


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class BaseVectorStore(ABC):
    """Minimal contract between the pipeline and any vector store."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert chunks with their embedding vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        query_text: str | None = None,
    ) -> list[SearchHit]:
        """Return the k most similar chunks, best first.

        ``query_text`` enables hybrid BM25 fusion when the backend supports it.
        """

    @abstractmethod
    def persist(self) -> None:
        """Flush the index so it survives process exit."""

    @abstractmethod
    def count(self) -> int:
        """Number of chunks in the index."""

    @abstractmethod
    def contains_doc(self, doc_id: str) -> bool:
        """True if any chunk from ``doc_id`` is already in the index."""


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _apply_diversity(hits: list[SearchHit], k: int, max_per_doc: int) -> list[SearchHit]:
    if max_per_doc <= 0:
        return hits[:k]
    counts: dict[str, int] = {}
    out: list[SearchHit] = []
    for hit in hits:
        doc = hit.chunk.doc_id
        if counts.get(doc, 0) >= max_per_doc:
            continue
        counts[doc] = counts.get(doc, 0) + 1
        out.append(hit)
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------- local


class LocalVectorStore(BaseVectorStore):
    """Numpy cosine-similarity store persisted as .npz + .json files."""

    def __init__(
        self,
        index_dir: Path,
        *,
        hybrid_enabled: bool = True,
        hybrid_alpha: float = 0.7,
        max_chunks_per_doc: int = 2,
    ):
        self.index_dir = index_dir
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_alpha = hybrid_alpha
        self.max_chunks_per_doc = max_chunks_per_doc
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self._bm25: BM25 | None = None
        self._doc_ids: set[str] = set()

    def _rebuild_lexical(self) -> None:
        self._bm25 = BM25([c.text for c in self.chunks]) if self.chunks else None
        self._doc_ids = {c.doc_id for c in self.chunks}

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise StoreError(f"{len(chunks)} chunks but {embeddings.shape[0]} embeddings")
        normalized = _normalize_rows(embeddings.astype(np.float32))
        self.vectors = normalized if self.vectors is None else np.vstack([self.vectors, normalized])
        self.chunks.extend(chunks)
        self._rebuild_lexical()
        logger.info("local add chunks=%d total=%d", len(chunks), len(self.chunks))

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        query_text: str | None = None,
    ) -> list[SearchHit]:
        if self.vectors is None or not self.chunks:
            raise StoreError("Index is empty. Run `rag ingest` first.")
        query = _normalize_vector(query_vector)
        dense = self.vectors @ query

        use_hybrid = (
            self.hybrid_enabled
            and query_text
            and self._bm25 is not None
            and 0.0 < self.hybrid_alpha < 1.0
        )
        if use_hybrid:
            lexical = np.asarray(self._bm25.scores(query_text), dtype=np.float32)
            scores = self.hybrid_alpha * _minmax(dense) + (1.0 - self.hybrid_alpha) * _minmax(lexical)
        else:
            scores = dense

        pool = max(k * 4, k + 10) if self.max_chunks_per_doc > 0 else k
        pool = min(pool, len(scores))
        top = np.argsort(-scores)[:pool]
        hits = [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in top]
        hits = _apply_diversity(hits, k=k, max_per_doc=self.max_chunks_per_doc)
        logger.debug(
            "local search k=%d hybrid=%s top_score=%.4f docs=%s",
            k,
            use_hybrid,
            hits[0].score if hits else 0.0,
            [h.chunk.doc_id for h in hits[:3]],
        )
        return hits

    def persist(self) -> None:
        if self.vectors is None:
            raise StoreError("Nothing to persist: store is empty")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_dir / _VECTORS_FILE, vectors=self.vectors)
        payload = [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "section": c.section,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (self.index_dir / _CHUNKS_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("local persisted chunks=%d path=%s", len(self.chunks), self.index_dir)

    def count(self) -> int:
        return len(self.chunks)

    def contains_doc(self, doc_id: str) -> bool:
        return doc_id in self._doc_ids

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        hybrid_enabled: bool = True,
        hybrid_alpha: float = 0.7,
        max_chunks_per_doc: int = 2,
    ) -> "LocalVectorStore":
        vectors_path = index_dir / _VECTORS_FILE
        chunks_path = index_dir / _CHUNKS_FILE
        if not vectors_path.exists() or not chunks_path.exists():
            raise StoreError(f"No index found in {index_dir}. Run `rag ingest` first.")
        store = cls(
            index_dir,
            hybrid_enabled=hybrid_enabled,
            hybrid_alpha=hybrid_alpha,
            max_chunks_per_doc=max_chunks_per_doc,
        )
        store.vectors = np.load(vectors_path)["vectors"]
        store.chunks = [Chunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
        if store.vectors.shape[0] != len(store.chunks):
            raise StoreError("Corrupt index: vector/chunk count mismatch. Re-run `rag ingest`.")
        store._rebuild_lexical()
        logger.info("local opened chunks=%d path=%s", len(store.chunks), index_dir)
        return store


# -------------------------------------------------------------------- qdrant


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed store. Chunk payloads live in the collection, so no
    local files are needed; ``persist`` is a no-op (Qdrant is durable)."""

    def __init__(
        self,
        url: str,
        collection: str,
        create: bool = False,
        upsert_batch_size: int = 256,
        *,
        hybrid_enabled: bool = True,
        hybrid_alpha: float = 0.7,
        max_chunks_per_doc: int = 2,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise StoreError(
                "The qdrant backend requires the qdrant-client package: pip install 'rag-base[qdrant]'"
            ) from exc
        self._models = __import__("qdrant_client.models", fromlist=["models"])
        self.client = QdrantClient(url=url)
        self.collection = collection
        self._next_id = 0
        self._upsert_batch_size = upsert_batch_size
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_alpha = hybrid_alpha
        self.max_chunks_per_doc = max_chunks_per_doc
        self._doc_ids: set[str] = set()
        if create:
            self._created = False
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
        elif not self.client.collection_exists(collection):
            raise StoreError(
                f"Qdrant collection '{collection}' does not exist at {url}. Run `rag ingest` first."
            )
        else:
            self._next_id = self.client.count(collection, exact=True).count
            self._refresh_doc_ids()

    def _refresh_doc_ids(self) -> None:
        try:
            doc_ids: set[str] = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=10_000,
                    offset=offset,
                    with_payload=["doc_id"],
                    with_vectors=False,
                )
                doc_ids.update(str((p.payload or {}).get("doc_id", "")) for p in points)
                if offset is None:
                    break
            doc_ids.discard("")
            self._doc_ids = doc_ids
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant doc_id cache refresh failed: %s", exc)
            self._doc_ids = set()

    def _ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(size=dim, distance=self._models.Distance.COSINE),
            )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise StoreError(f"{len(chunks)} chunks but {embeddings.shape[0]} embeddings")
        self._ensure_collection(embeddings.shape[1])
        points = [
            self._models.PointStruct(
                id=self._next_id + i,
                vector=embeddings[i].tolist(),
                payload={
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "section": c.section,
                    "metadata": c.metadata,
                },
            )
            for i, c in enumerate(chunks)
        ]
        self._next_id += len(chunks)
        for start in range(0, len(points), self._upsert_batch_size):
            batch = points[start : start + self._upsert_batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=True,
            )
            logger.debug("qdrant upsert batch=%d-%d total=%d", start + 1, start + len(batch), len(points))
        self._doc_ids.update(c.doc_id for c in chunks)
        logger.info("qdrant add points=%d total=%d", len(chunks), self._next_id)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        query_text: str | None = None,
    ) -> list[SearchHit]:
        pool = max(k * 4, k + 10) if (self.max_chunks_per_doc > 0 or self.hybrid_enabled) else k
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.astype(np.float32).reshape(-1).tolist(),
            limit=pool,
            with_payload=True,
        )
        candidates: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            candidates.append(
                SearchHit(
                    chunk=Chunk(
                        doc_id=payload.get("doc_id", ""),
                        chunk_id=payload.get("chunk_id", ""),
                        text=payload.get("text", ""),
                        section=payload.get("section", ""),
                        metadata=payload.get("metadata", {}),
                    ),
                    score=float(point.score),
                )
            )

        use_hybrid = (
            self.hybrid_enabled
            and query_text
            and candidates
            and 0.0 < self.hybrid_alpha < 1.0
        )
        if use_hybrid:
            bm25 = BM25([h.chunk.text for h in candidates])
            lexical = np.asarray(bm25.scores(query_text), dtype=np.float32)
            dense = np.asarray([h.score for h in candidates], dtype=np.float32)
            fused = self.hybrid_alpha * _minmax(dense) + (1.0 - self.hybrid_alpha) * _minmax(lexical)
            order = np.argsort(-fused)
            candidates = [
                SearchHit(chunk=candidates[i].chunk, score=float(fused[i])) for i in order
            ]

        hits = _apply_diversity(candidates, k=k, max_per_doc=self.max_chunks_per_doc)
        logger.debug(
            "qdrant search k=%d hybrid=%s top_score=%.4f docs=%s",
            k,
            use_hybrid,
            hits[0].score if hits else 0.0,
            [h.chunk.doc_id for h in hits[:3]],
        )
        return hits

    def persist(self) -> None:
        pass

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def contains_doc(self, doc_id: str) -> bool:
        if doc_id in self._doc_ids:
            return True
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._models.Filter(
                    must=[self._models.FieldCondition(key="doc_id", match=self._models.MatchValue(value=doc_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            found = bool(points)
            if found:
                self._doc_ids.add(doc_id)
            return found
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant contains_doc failed doc=%s error=%s", doc_id, exc)
            return False


# ------------------------------------------------------------------- factory

_BACKENDS = ("local", "qdrant")


def _check_backend(config: RagConfig) -> None:
    if config.vector_backend not in _BACKENDS:
        raise StoreError(
            f"Unknown vector backend '{config.vector_backend}'. Choose from: {', '.join(_BACKENDS)}"
        )


def create_store(config: RagConfig) -> BaseVectorStore:
    """A fresh, empty store for (re)ingestion. Any existing index for the
    configured backend is discarded."""
    _check_backend(config)
    if config.vector_backend == "qdrant":
        logger.info("qdrant create url=%s collection=%s", config.qdrant_url, config.qdrant_collection)
        return QdrantVectorStore(
            config.qdrant_url,
            config.qdrant_collection,
            create=True,
            upsert_batch_size=config.qdrant_upsert_batch_size,
            hybrid_enabled=config.hybrid_enabled,
            hybrid_alpha=config.hybrid_alpha,
            max_chunks_per_doc=config.max_chunks_per_doc,
        )
    logger.info("local create path=%s", config.index_dir)
    return LocalVectorStore(
        config.index_dir,
        hybrid_enabled=config.hybrid_enabled,
        hybrid_alpha=config.hybrid_alpha,
        max_chunks_per_doc=config.max_chunks_per_doc,
    )


def open_store(config: RagConfig) -> BaseVectorStore:
    """Open the existing index for querying. Raises StoreError if absent."""
    _check_backend(config)
    if config.vector_backend == "qdrant":
        logger.debug("qdrant open url=%s collection=%s", config.qdrant_url, config.qdrant_collection)
        return QdrantVectorStore(
            config.qdrant_url,
            config.qdrant_collection,
            create=False,
            hybrid_enabled=config.hybrid_enabled,
            hybrid_alpha=config.hybrid_alpha,
            max_chunks_per_doc=config.max_chunks_per_doc,
        )
    logger.debug("local open path=%s", config.index_dir)
    return LocalVectorStore.open(
        config.index_dir,
        hybrid_enabled=config.hybrid_enabled,
        hybrid_alpha=config.hybrid_alpha,
        max_chunks_per_doc=config.max_chunks_per_doc,
    )
