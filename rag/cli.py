"""Command-line interface: rag scrape | ingest | ask | chat | eval | info."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .llm import LLMError, create_llm
from .store import StoreError, open_store

app = typer.Typer(add_completion=False, help="A basic local RAG over Marvel/DC Fandom data, with metrics.")
console = Console()


def _client_and_config():
    config = load_config()
    try:
        return config, create_llm(config)
    except LLMError as exc:
        _fail(str(exc))


def _fail(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


@app.command()
def scrape(
    wiki: Optional[str] = typer.Option(None, help="Only scrape one wiki: 'marvel' or 'dc'."),
    character: Optional[str] = typer.Option(None, help="Scrape a single extra page title (requires --wiki)."),
):
    """Download Marvel/DC character articles from Fandom into data/corpus/."""
    from .scrape import WIKIS, scrape_all, slugify

    config = load_config()
    if wiki and wiki not in WIKIS:
        _fail(f"Unknown wiki '{wiki}'. Choose from: {', '.join(WIKIS)}")

    extra = []
    if character:
        if not wiki:
            _fail("--character requires --wiki (marvel or dc)")
        extra = [(wiki, character, slugify(character))]

    def report(result):
        if result.ok:
            console.print(f"  [green]ok[/green]  {result.wiki:6s} {result.title}  ({result.chars:,} chars)")
        else:
            console.print(f"  [red]FAIL[/red] {result.wiki:6s} {result.title}: {result.error}")

    console.print(f"Scraping into [bold]{config.corpus_dir}[/bold] ...")
    if character:
        results = scrape_all(config, only_wiki="__none__", extra=extra, progress=report)
    else:
        results = scrape_all(config, only_wiki=wiki, extra=extra, progress=report)
    ok = sum(1 for r in results if r.ok)
    console.print(f"\nDone: [green]{ok} scraped[/green], [red]{len(results) - ok} failed[/red]. "
                  f"Manifest: {config.corpus_dir / 'manifest.json'}")


@app.command()
def ingest():
    """Chunk and embed the corpus, building the local vector index."""
    from .ingest import build_index

    config, client = _client_and_config()
    try:
        client.check_ready([config.embed_model])
    except LLMError as exc:
        _fail(str(exc))

    console.print(f"Indexing corpus from [bold]{config.corpus_dir}[/bold] ...")

    def report(doc_id, n_chunks):
        console.print(f"  chunked {doc_id}: {n_chunks} chunks")

    try:
        with console.status("Embedding chunks (this can take a couple of minutes)..."):
            stats = build_index(config, client, progress=None if console.is_terminal else report)
    except (FileNotFoundError, LLMError, StoreError) as exc:
        _fail(str(exc))
    console.print(
        f"[green]Indexed {stats.documents} documents into {stats.chunks} chunks[/green] "
        f"-> {stats.backend} backend ({stats.location})"
    )


def _load_pipeline():
    from .pipeline import RagPipeline

    config, client = _client_and_config()
    try:
        client.check_ready([config.embed_model, config.chat_model])
        store = open_store(config)
    except (LLMError, StoreError) as exc:
        _fail(str(exc))
    return config, client, RagPipeline(config, client, store)


def _print_answer(result, show_sources: bool):
    console.print(Panel(result.answer, title="Answer", border_style="cyan"))
    if getattr(result, "fallback_used", False):
        doc = getattr(result, "fallback_doc", "") or "unknown page"
        console.print(
            f"[yellow]Low retrieval score — fetched and indexed from Fandom:[/yellow] {doc}"
        )
    if show_sources:
        table = Table(title="Retrieved sources", show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("Document")
        table.add_column("Section")
        table.add_column("Score", justify="right")
        for s in result.sources:
            table.add_row(str(s["rank"]), s["title"] or s["doc"], s["section"] or "-", f"{s['score']:.3f}")
        console.print(table)
    console.print(
        f"[dim]retrieval {result.retrieval_seconds * 1000:.0f} ms · total {result.total_seconds:.1f} s[/dim]"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    k: Optional[int] = typer.Option(None, help="Number of chunks to retrieve (default from config)."),
    sources: bool = typer.Option(True, help="Show retrieved sources."),
):
    """Ask a single question against the indexed corpus."""
    config, client, pipeline = _load_pipeline()
    with console.status("Thinking..."):
        result = pipeline.answer(question, k=k)
    _print_answer(result, sources)


@app.command()
def chat():
    """Interactive question loop (each question is independent)."""
    config, client, pipeline = _load_pipeline()
    console.print("[bold]RAG chat[/bold] — ask about Marvel/DC characters. Ctrl-D or 'exit' to quit.\n")
    while True:
        try:
            question = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        with console.status("Thinking..."):
            result = pipeline.answer(question)
        _print_answer(result, show_sources=False)
    console.print("bye")


@app.command("eval")
def eval_cmd(
    limit: Optional[int] = typer.Option(None, help="Only run the first N questions."),
    category: Optional[str] = typer.Option(None, help="Only run questions of one category."),
):
    """Run the evaluation set and report retrieval + generation metrics."""
    from .evaluate import run_evaluation

    config, client, pipeline = _load_pipeline()

    def report(result):
        marker = "[green]ok[/green]" if not result.scores.get("abstained") else "[yellow]abstained[/yellow]"
        if not result.item.answerable:
            marker = "[green]ok[/green]" if result.scores.get("abstention_correct") else "[red]hallucinated[/red]"
        console.print(f"  {marker} [{result.item.category}] {result.item.question[:70]}")

    console.print(f"Evaluating with judge model [bold]{config.effective_judge_model()}[/bold] ...")
    try:
        report_data = run_evaluation(config, client, pipeline, limit=limit, category=category, progress=report)
    except (FileNotFoundError, ValueError, LLMError) as exc:
        _fail(str(exc))

    def add_metric_rows(table: Table, agg: dict):
        labels = {
            "hit_rate": "Hit Rate@k",
            "precision": "Precision@k",
            "recall": "Recall@k",
            "mrr": "MRR",
            "ndcg": "nDCG@k",
            "token_f1": "Token F1",
            "faithfulness": "Faithfulness (1-5)",
            "relevance": "Answer relevance (1-5)",
            "correctness": "Correctness (1-5)",
            "abstention_correct": "Abstention accuracy",
            "latency_mean_s": "Latency mean (s)",
            "latency_p95_s": "Latency p95 (s)",
            "retrieval_mean_s": "Retrieval mean (s)",
        }
        for key, label in labels.items():
            if key in agg:
                table.add_row(label, str(agg[key]))

    overall = Table(title=f"Overall ({report_data['overall']['n']} questions)")
    overall.add_column("Metric")
    overall.add_column("Value", justify="right")
    add_metric_rows(overall, report_data["overall"])
    console.print(overall)

    by_cat = Table(title="By category")
    by_cat.add_column("Category")
    by_cat.add_column("n", justify="right")
    for key in ["hit_rate", "mrr", "token_f1", "correctness", "abstention_correct"]:
        by_cat.add_column(key, justify="right")
    for cat, agg in report_data["by_category"].items():
        by_cat.add_row(
            cat,
            str(agg.get("n", "")),
            *[str(agg.get(k, "-")) for k in ["hit_rate", "mrr", "token_f1", "correctness", "abstention_correct"]],
        )
    console.print(by_cat)
    console.print(f"Full per-question results: [bold]{config.eval_results_path}[/bold]")


@app.command()
def info():
    """Show current configuration and index status."""
    config, client = _client_and_config()
    table = Table(title="rag-base configuration")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("LLM provider", config.llm_provider)
    table.add_row("Chat model", config.chat_model)
    table.add_row("Embed model", config.embed_model)
    table.add_row("Judge model", config.effective_judge_model())
    table.add_row("Top-k", str(config.top_k))
    table.add_row("Chunk size (chars)", str(config.chunk_chars))
    table.add_row("Embed cache size", str(config.embed_cache_size))
    table.add_row(
        "Fandom fallback",
        (
            f"on (min score {config.fallback_min_score})"
            if config.fallback_enabled
            else "off"
        ),
    )
    table.add_row("Corpus dir", str(config.corpus_dir))
    table.add_row("Vector backend", config.vector_backend)
    if config.vector_backend == "qdrant":
        table.add_row("Qdrant", f"{config.qdrant_url} / {config.qdrant_collection}")
    else:
        table.add_row("Index dir", str(config.index_dir))
    try:
        store = open_store(config)
        table.add_row("Index", f"{store.count()} chunks")
    except StoreError:
        table.add_row("Index", "[red]not built (run `rag ingest`)[/red]")
    try:
        models = client.available_models()
        table.add_row("Backend", f"{client.describe()} — up ({len(models)} models)")
    except LLMError:
        table.add_row("Backend", f"{client.describe()} — [red]unreachable[/red]")
    console.print(table)


if __name__ == "__main__":
    app()
