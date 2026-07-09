"""Central logging configuration for the RAG pipeline.

Logs are diagnostic output (retries, latencies, retrieval scores, fallback
decisions) and are kept separate from the user-facing Rich output in
``cli.py``: they go to stderr (and optionally a file), so piping ``rag ask``
stdout stays clean.

Call :func:`configure_logging` once at process start (the CLI does this in its
callback). Every module obtains a logger via :func:`get_logger`; because all
loggers live under the ``rag`` namespace, a single configuration governs them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .config import RagConfig

_ROOT_LOGGER_NAME = "rag"
_configured = False


def configure_logging(config: RagConfig, *, force: bool = False) -> None:
    """Configure the ``rag`` logger tree from ``config``.

    Idempotent: repeated calls are ignored unless ``force`` is set (handy for
    tests or when the level changes mid-process).
    """
    global _configured
    if _configured and not force:
        return

    level = logging.getLevelName(config.log_level.strip().upper())
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console_handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_path=False,
        markup=False,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(name)s — %(message)s", datefmt="[%X]"))
    root.addHandler(console_handler)

    if config.log_file:
        file_path = Path(config.log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
        )
        root.addHandler(file_handler)

    _configured = True
    root.debug("logging level=%s file=%s", config.log_level, config.log_file or "-")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Pass ``__name__`` from each module."""
    return logging.getLogger(name)
