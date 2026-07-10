"""Scrape Marvel and DC character articles from Fandom wikis.

Uses each wiki's MediaWiki API (``action=parse``) rather than raw HTML pages,
then converts the article HTML to clean markdown with the stdlib HTML parser:
navigation, galleries, and reference markup are dropped. Infobox key/value
pairs are extracted into a ``## Quick Facts`` section (aliases, real names,
affiliations) before the body. Section headings are preserved so downstream
chunking follows article structure.

Robustness: retries with exponential backoff, redirect following, per-page
skip-and-report on failure, a polite delay between requests, an identifying
User-Agent, and a ``manifest.json`` describing everything that was fetched.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import requests

from .config import RagConfig
from .logging_config import get_logger

logger = get_logger(__name__)

USER_AGENT = "rag-base/0.1 (educational RAG demo; https://github.com/)"

WIKIS = {
    "marvel": "https://marvel.fandom.com",
    "dc": "https://dc.fandom.com",
}

CHARACTERS: dict[str, list[tuple[str, str]]] = {
    "marvel": [
        ("Peter Parker (Earth-616)", "spider_man"),
        ("Anthony Stark (Earth-616)", "iron_man"),
        ("Steven Rogers (Earth-616)", "captain_america"),
        ("Thor Odinson (Earth-616)", "thor"),
        ("Bruce Banner (Earth-616)", "hulk"),
        ("Wanda Maximoff (Earth-616)", "scarlet_witch"),
        ("Stephen Strange (Earth-616)", "doctor_strange"),
        ("James Howlett (Earth-616)", "wolverine"),
        ("Ororo Munroe (Earth-616)", "storm"),
        ("Scott Summers (Earth-616)", "cyclops"),
        ("Jean Grey (Earth-616)", "jean_grey"),
        ("Matthew Murdock (Earth-616)", "daredevil"),
        ("Carol Danvers (Earth-616)", "captain_marvel"),
        ("T'Challa (Earth-616)", "black_panther"),
        ("Natalia Romanova (Earth-616)", "black_widow"),
        ("Wade Wilson (Earth-616)", "deadpool"),
        ("Clinton Barton (Earth-616)", "hawkeye"),
        ("Peter Quill (Earth-616)", "star_lord"),
        ("Thanos (Earth-616)", "thanos"),
        ("Victor von Doom (Earth-616)", "doctor_doom"),
        ("Norman Osborn (Earth-616)", "green_goblin"),
        ("Max Eisenhardt (Earth-616)", "magneto"),
        ("Loki Laufeyson (Earth-616)", "loki"),
        ("Johann Shmidt (Earth-616)", "red_skull"),
        ("Avengers (Earth-616)", "avengers"),
        ("X-Men (Earth-616)", "x_men"),
        ("Fantastic Four (Earth-616)", "fantastic_four"),
    ],
    "dc": [
        ("Bruce Wayne (New Earth)", "batman"),
        ("Kal-El (New Earth)", "superman"),
        ("Diana of Themyscira (New Earth)", "wonder_woman"),
        ("Barry Allen (New Earth)", "flash"),
        ("Hal Jordan (New Earth)", "green_lantern"),
        ("Orin (New Earth)", "aquaman"),
        ("J'onn J'onzz (New Earth)", "martian_manhunter"),
        ("Oliver Queen (New Earth)", "green_arrow"),
        ("Dick Grayson (New Earth)", "nightwing"),
        ("Barbara Gordon (New Earth)", "barbara_gordon"),
        ("Victor Stone (New Earth)", "cyborg"),
        ("Billy Batson (New Earth)", "shazam"),
        ("Kara Zor-El (New Earth)", "supergirl"),
        ("Selina Kyle (New Earth)", "catwoman"),
        ("Joker (New Earth)", "joker"),
        ("Alexander Luthor (New Earth)", "lex_luthor"),
        ("Harleen Quinzel (New Earth)", "harley_quinn"),
        ("Slade Wilson (New Earth)", "deathstroke"),
        ("Uxas (New Earth)", "darkseid"),
        ("Bane (New Earth)", "bane"),
        ("Ra's al Ghul (New Earth)", "ras_al_ghul"),
        ("Justice League of America (New Earth)", "justice_league"),
        ("Teen Titans (New Earth)", "teen_titans"),
        ("Suicide Squad (New Earth)", "suicide_squad"),
    ],
}

_DROP_SECTIONS = {
    "references", "external links", "links", "links and references", "see also",
    "notes", "trivia", "gallery", "recommended reading", "related", "footnotes",
    "discover and discuss",
}


@dataclass
class ScrapeResult:
    wiki: str
    title: str
    slug: str
    ok: bool
    path: str = ""
    url: str = ""
    chars: int = 0
    error: str = ""


class _InfoboxExtractor(HTMLParser):
    """Pull key/value pairs from Fandom character infobox tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.facts: list[tuple[str, str]] = []
        self._in_infobox = 0
        self._in_row = False
        self._in_th = False
        self._in_td = False
        self._th_buf: list[str] = []
        self._td_buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        classes = attr_map.get("class") or ""
        if tag == "table" and "infobox" in classes:
            self._in_infobox += 1
            return
        if not self._in_infobox:
            return
        if tag in {"script", "style", "sup"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "tr":
            self._in_row = True
            self._th_buf = []
            self._td_buf = []
        elif tag == "th" and self._in_row:
            self._in_th = True
        elif tag == "td" and self._in_row:
            self._in_td = True
        elif tag == "br" and self._in_td:
            self._td_buf.append(", ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "sup"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "table" and self._in_infobox:
            self._in_infobox -= 1
            return
        if not self._in_infobox or self._skip_depth:
            return
        if tag == "th":
            self._in_th = False
        elif tag == "td":
            self._in_td = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
            key = re.sub(r"\s+", " ", "".join(self._th_buf)).strip().rstrip(":")
            value = re.sub(r"\s+", " ", "".join(self._td_buf)).strip()
            value = re.sub(r"\s*,\s*,+", ",", value).strip(" ,")
            if key and value and len(value) < 500:
                self.facts.append((key, value))

    def handle_data(self, data):
        if not self._in_infobox or self._skip_depth:
            return
        if self._in_th:
            self._th_buf.append(data)
        elif self._in_td:
            self._td_buf.append(data)


_INFOBOX_KEYS = {
    "real name",
    "current alias",
    "aliases",
    "alias",
    "identity",
    "affiliation",
    "affiliations",
    "relatives",
    "base of operations",
    "status",
    "citizenship",
    "marital status",
    "occupation",
    "gender",
    "height",
    "weight",
    "eyes",
    "hair",
    "creators",
    "first",
    "first appearance",
    "place of birth",
    "origin",
    "team affiliations",
    "group affiliation",
    "powers",
    "abilities",
}


def extract_infobox_facts(html: str) -> list[tuple[str, str]]:
    """Return curated (label, value) pairs from character infoboxes."""
    parser = _InfoboxExtractor()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for key, value in parser.facts:
        key_norm = key.lower()
        if key_norm not in _INFOBOX_KEYS and not any(k in key_norm for k in ("alias", "name", "affiliation")):
            continue
        if key_norm in seen:
            continue
        seen.add(key_norm)
        out.append((key, value))
    return out


def _facts_markdown(facts: list[tuple[str, str]]) -> str:
    if not facts:
        return ""
    lines = ["## Quick Facts", ""]
    for key, value in facts:
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


class _HtmlToMarkdown(HTMLParser):
    """Convert MediaWiki article HTML into plain markdown text."""

    _SKIP_TAGS = {"aside", "table", "figure", "script", "style", "sup", "nav", "audio", "video"}
    _SKIP_CLASS_RE = re.compile(
        r"toc|navbox|infobox|gallery|mw-editsection|reference|quote-source|printfooter|catlinks|wikia-gallery"
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._skip_stack: list[str] = []
        self._heading: str | None = None
        self._in_list_item = False

    def _flush(self) -> None:
        text = "".join(self._buffer).strip()
        self._buffer = []
        if text:
            self.blocks.append(text)

    def _should_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in self._SKIP_TAGS:
            return True
        attr_map = dict(attrs)
        blob = f"{attr_map.get('class') or ''} {attr_map.get('id') or ''}"
        return bool(self._SKIP_CLASS_RE.search(blob))

    def handle_starttag(self, tag, attrs):
        if self._skip_depth:
            if tag == self._skip_stack[-1]:
                self._skip_stack.append(tag)
                self._skip_depth += 1
            return
        if self._should_skip(tag, attrs):
            self._skip_stack.append(tag)
            self._skip_depth = 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5"):
            self._flush()
            self._heading = "#" * min(int(tag[1]) , 4)
        elif tag == "p":
            self._flush()
        elif tag == "li":
            self._flush()
            self._in_list_item = True
        elif tag == "br":
            self._buffer.append("\n")

    def handle_endtag(self, tag):
        if self._skip_depth:
            if self._skip_stack and tag == self._skip_stack[-1]:
                self._skip_stack.pop()
                self._skip_depth -= 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5") and self._heading is not None:
            text = "".join(self._buffer).strip()
            self._buffer = []
            if text:
                self.blocks.append(f"{self._heading} {text}")
            self._heading = None
        elif tag == "p":
            self._flush()
        elif tag == "li":
            text = "".join(self._buffer).strip()
            self._buffer = []
            if text:
                self.blocks.append(f"- {text}")
            self._in_list_item = False

    def handle_data(self, data):
        if not self._skip_depth:
            self._buffer.append(data)

    def close(self):
        super().close()
        self._flush()


def html_to_markdown(html: str, max_chars: int) -> str:
    facts = extract_infobox_facts(html)
    facts_md = _facts_markdown(facts)

    parser = _HtmlToMarkdown()
    parser.feed(html)
    parser.close()

    lines: list[str] = []
    skipping_section = False
    for block in parser.blocks:
        m = re.match(r"^(#{1,4})\s+(.*)$", block)
        if m:
            title = re.sub(r"\[.*?\]", "", m.group(2)).strip()
            skipping_section = title.lower() in _DROP_SECTIONS
            if not skipping_section:
                lines.append(f"{m.group(1)} {title}")
            continue
        if not skipping_section:
            cleaned = re.sub(r"\[\d+\]", "", block)
            cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
            if cleaned:
                lines.append(cleaned)

    body = "\n\n".join(lines)
    if facts_md:
        text = f"{facts_md}\n\n{body}".strip()
    else:
        text = body
    if len(text) > max_chars:
        cut = text.rfind("\n\n", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars]
    return text.strip()


def _fetch_page_html(session: requests.Session, base_url: str, title: str, retries: int = 3) -> tuple[str, str]:
    """Return (resolved_title, article_html) for a wiki page title."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": "1",
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
            parsed = data["parse"]
            return parsed.get("title", title), parsed["text"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < retries - 1:
                backoff = 2**attempt
                logger.warning(
                    "fetch title=%s attempt=%d/%d backoff_s=%d error=%s",
                    title,
                    attempt + 1,
                    retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
    logger.error("fetch title=%s attempts=%d error=%s", title, retries, last_error)
    raise RuntimeError(f"failed to fetch '{title}': {last_error}")


def scrape_page(
    session: requests.Session,
    config: RagConfig,
    wiki: str,
    title: str,
    slug: str,
) -> ScrapeResult:
    base_url = WIKIS[wiki]
    url = f"{base_url}/wiki/{title.replace(' ', '_')}"
    try:
        resolved_title, html = _fetch_page_html(session, base_url, title)
        body = html_to_markdown(html, max_chars=config.max_doc_chars)
        if len(body) < 500:
            raise RuntimeError(f"extracted only {len(body)} chars — page is likely empty or a redirect stub")
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
        logger.info("scrape ok wiki=%s title=%s resolved=%s chars=%d", wiki, title, resolved_title, len(body))
        return ScrapeResult(wiki=wiki, title=resolved_title, slug=slug, ok=True, path=str(path), url=url, chars=len(body))
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape fail wiki=%s title=%s error=%s", wiki, title, exc)
        return ScrapeResult(wiki=wiki, title=title, slug=slug, ok=False, url=url, error=str(exc))


def scrape_all(
    config: RagConfig,
    only_wiki: str | None = None,
    extra: list[tuple[str, str, str]] | None = None,
    progress=None,
) -> list[ScrapeResult]:
    """Scrape the default character set (plus any ``extra`` (wiki, title, slug)
    entries). Returns one result per page; failures don't abort the run."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    jobs: list[tuple[str, str, str]] = []
    for wiki, entries in CHARACTERS.items():
        if only_wiki and wiki != only_wiki:
            continue
        jobs.extend((wiki, title, slug) for title, slug in entries)
    jobs.extend(extra or [])

    logger.info("scrape pages=%d wiki=%s", len(jobs), only_wiki or "all")
    results: list[ScrapeResult] = []
    for wiki, title, slug in jobs:
        result = scrape_page(session, config, wiki, title, slug)
        results.append(result)
        if progress:
            progress(result)
        time.sleep(config.scrape_delay_seconds)

    _update_manifest(config, results)
    ok = sum(1 for r in results if r.ok)
    logger.info("scrape done ok=%d failed=%d", ok, len(results) - ok)
    return results


def _update_manifest(config: RagConfig, results: list[ScrapeResult]) -> None:
    """Merge this run's results into the manifest (keyed by wiki+slug), so
    partial scrapes don't clobber the record of earlier runs."""
    manifest_path = config.corpus_dir / "manifest.json"
    pages: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            for page in json.loads(manifest_path.read_text(encoding="utf-8")).get("pages", []):
                pages[f"{page['wiki']}__{page['slug']}"] = page
        except (ValueError, KeyError):
            pass
    for r in results:
        pages[f"{r.wiki}__{r.slug}"] = vars(r)
    ordered = sorted(pages.values(), key=lambda p: (p["wiki"], p["slug"]))
    manifest = {
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pages": ordered,
        "ok": sum(1 for p in ordered if p["ok"]),
        "failed": sum(1 for p in ordered if not p["ok"]),
    }
    config.corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.debug("manifest pages=%d ok=%d failed=%d", len(ordered), manifest["ok"], manifest["failed"])


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
