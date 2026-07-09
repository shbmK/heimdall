"""Live Fandom MediaWiki fallback when local retrieval scores are too low.

Searches Marvel and DC Fandom, fetches the best matching page, writes it into
the corpus, appends its chunks to the vector store, and returns metadata so
the pipeline can re-retrieve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .chunking import chunk_document
from .config import RagConfig
from .llm import LLMProvider
from .scrape import (
    USER_AGENT,
    WIKIS,
    ScrapeResult,
    _fetch_page_html,
    _update_manifest,
    html_to_markdown,
    slugify,
)
from .store import BaseVectorStore


@dataclass
class FallbackResult:
    """Outcome of a fallback attempt."""

    ok: bool
    doc_id: str = ""
    title: str = ""
    wiki: str = ""
    url: str = ""
    chunks_added: int = 0
    skipped_existing: bool = False
    error: str = ""


def _search_wiki(
    session: requests.Session,
    wiki: str,
    query: str,
    limit: int,
    retries: int = 3,
) -> list[tuple[str, str, float]]:
    """Return (wiki, title, score) hits from MediaWiki list=search."""
    base_url = WIKIS[wiki]
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srprop": "snippet",
        "format": "json",
        "formatversion": "2",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(f"{base_url}/api.php", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise ValueError(data["error"].get("info", "unknown API error"))
            hits = []
            for i, item in enumerate(data.get("query", {}).get("search", [])):
                title = item.get("title", "")
                if not title:
                    continue
                # MediaWiki search order is relevance; assign descending rank scores.
                hits.append((wiki, title, float(limit - i)))
            return hits
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Fandom search failed on {wiki}: {last_error}")


def _title_match_score(query: str, title: str) -> float:
    """Simple token overlap for breaking ties across wikis."""
    q_tokens = {t for t in query.lower().split() if len(t) > 1}
    if not q_tokens:
        return 0.0
    t_tokens = set(title.lower().replace("(", " ").replace(")", " ").split())
    return len(q_tokens & t_tokens) / len(q_tokens)


def _pick_best_hit(
    hits: list[tuple[str, str, float]], query: str = ""
) -> tuple[str, str] | None:
    if not hits:
        return None
    wiki, title, _ = max(
        hits,
        key=lambda h: (h[2], _title_match_score(query, h[1])),
    )
    return wiki, title


def _doc_id_for(wiki: str, title: str) -> str:
    return f"{wiki}__{slugify(title)}.md"


def _write_corpus_page(
    config: RagConfig,
    wiki: str,
    resolved_title: str,
    slug: str,
    body: str,
    url: str,
) -> ScrapeResult:
    doc_id = f"{wiki}__{slug}.md"
    path = config.corpus_dir / doc_id
    header = (
        f"---\n"
        f"title: {resolved_title}\n"
        f"universe: {'Marvel' if wiki == 'marvel' else 'DC'}\n"
        f"source: {url}\n"
        f"scraped: {time.strftime('%Y-%m-%d')}\n"
        f"---\n\n"
        f"# {resolved_title}\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body + "\n", encoding="utf-8")
    return ScrapeResult(
        wiki=wiki,
        title=resolved_title,
        slug=slug,
        ok=True,
        path=str(path),
        url=url,
        chars=len(body),
    )


def fetch_and_ingest(
    config: RagConfig,
    client: LLMProvider,
    store: BaseVectorStore,
    query: str,
) -> FallbackResult:
    """Search Fandom for ``query``, ingest the best page, append to the store."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    all_hits: list[tuple[str, str, float]] = []
    for wiki in WIKIS:
        try:
            all_hits.extend(
                _search_wiki(session, wiki, query, limit=config.fallback_max_results)
            )
        except RuntimeError:
            continue  # one wiki down should not abort the other
        time.sleep(config.scrape_delay_seconds)

    best = _pick_best_hit(all_hits, query=query)
    if best is None:
        return FallbackResult(ok=False, error="no Fandom search results")

    wiki, title = best
    slug = slugify(title)
    doc_id = _doc_id_for(wiki, title)
    corpus_path = config.corpus_dir / doc_id

    if corpus_path.exists():
        return FallbackResult(
            ok=False,
            doc_id=doc_id,
            title=title,
            wiki=wiki,
            skipped_existing=True,
            error=f"corpus already has {doc_id}",
        )

    base_url = WIKIS[wiki]
    url = f"{base_url}/wiki/{title.replace(' ', '_')}"
    try:
        resolved_title, html = _fetch_page_html(session, base_url, title)
        body = html_to_markdown(html, max_chars=config.max_doc_chars)
        if len(body) < 500:
            return FallbackResult(
                ok=False,
                doc_id=doc_id,
                title=title,
                wiki=wiki,
                url=url,
                error=f"extracted only {len(body)} chars — page is likely empty",
            )

        # Prefer slug from resolved title so filenames stay stable after redirects.
        slug = slugify(resolved_title)
        doc_id = f"{wiki}__{slug}.md"
        corpus_path = config.corpus_dir / doc_id
        if corpus_path.exists():
            return FallbackResult(
                ok=False,
                doc_id=doc_id,
                title=resolved_title,
                wiki=wiki,
                url=url,
                skipped_existing=True,
                error=f"corpus already has {doc_id}",
            )

        result = _write_corpus_page(config, wiki, resolved_title, slug, body, url)
        _update_manifest(config, [result])

        chunks = chunk_document(
            doc_id=doc_id,
            title=resolved_title,
            text=body,
            max_chars=config.chunk_chars,
            overlap_chars=config.chunk_overlap_chars,
            metadata={
                "title": resolved_title,
                "universe": "Marvel" if wiki == "marvel" else "DC",
                "source": url,
            },
        )
        if not chunks:
            return FallbackResult(
                ok=False,
                doc_id=doc_id,
                title=resolved_title,
                wiki=wiki,
                url=url,
                error="no chunks produced from fetched page",
            )

        embeddings = client.embed([c.text for c in chunks])
        store.add(chunks, embeddings)
        store.persist()

        return FallbackResult(
            ok=True,
            doc_id=doc_id,
            title=resolved_title,
            wiki=wiki,
            url=url,
            chunks_added=len(chunks),
        )
    except Exception as exc:  # noqa: BLE001 — fallback must not crash ask/chat
        return FallbackResult(
            ok=False,
            doc_id=doc_id,
            title=title,
            wiki=wiki,
            url=url,
            error=str(exc),
        )
