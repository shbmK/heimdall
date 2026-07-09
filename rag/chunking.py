"""Markdown-aware document chunking.

Documents are split along markdown headings first, so chunks never straddle
two unrelated sections. Long sections are further split on paragraph
boundaries with a sliding character overlap. Each chunk is prefixed with its
document title and section path, which materially improves embedding recall
for questions that mention a character by name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section: str = ""
    metadata: dict = field(default_factory=dict)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, body) pairs. Text before the first
    heading gets an empty section title."""
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if current_lines and any(l.strip() for l in current_lines):
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines and any(l.strip() for l in current_lines):
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack paragraphs into pieces of at most ``max_chars``,
    carrying a tail of the previous piece as overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        # A single paragraph longer than the budget is hard-split on sentences.
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            head, para = para[:cut].strip(), para[cut:].strip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = f"...{overlap}\n\n{para}".strip() if overlap else para
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 250,
    metadata: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = _split_sections(text)
    for section_title, body in sections:
        for piece in _split_long_text(body, max_chars=max_chars, overlap_chars=overlap_chars):
            header = f"[{title}]" + (f" — {section_title}" if section_title else "")
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    text=f"{header}\n{piece}",
                    section=section_title,
                    metadata=dict(metadata or {}),
                )
            )
    logger.debug("chunk doc=%s sections=%d chunks=%d max_chars=%d", doc_id, len(sections), len(chunks), max_chars)
    return chunks
