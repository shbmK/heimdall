FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY rag ./rag
RUN pip install ".[qdrant]"

# ship the curated corpus and eval set so the image works out of the box
COPY data/corpus ./data/corpus
COPY data/eval ./data/eval
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# run as an unprivileged user and give it ownership of the workdir
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV RAG_LLM_PROVIDER=ollama \
    RAG_OLLAMA_HOST=http://ollama:11434 \
    RAG_VECTOR_BACKEND=qdrant \
    RAG_QDRANT_URL=http://qdrant:6333 \
    RAG_AUTO_INGEST=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# stay up so you can `docker compose exec rag rag ask "..."`
CMD ["tail", "-f", "/dev/null"]
