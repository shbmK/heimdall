"""Evaluation runner: executes the eval set through the pipeline and
aggregates metrics overall and per category."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import metrics
from .config import RagConfig
from .llm import LLMProvider
from .logging_config import get_logger
from .pipeline import RagPipeline

logger = get_logger(__name__)


@dataclass
class EvalItem:
    id: str
    question: str
    category: str
    answerable: bool
    relevant_docs: list[str] = field(default_factory=list)
    reference_answer: str = ""


@dataclass
class QuestionResult:
    item: EvalItem
    answer: str
    retrieved_docs: list[str]
    scores: dict
    retrieval_seconds: float
    total_seconds: float


def load_eval_set(path: Path) -> list[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for i, entry in enumerate(raw["questions"]):
        items.append(
            EvalItem(
                id=entry.get("id", f"q{i + 1}"),
                question=entry["question"],
                category=entry.get("category", "uncategorized"),
                answerable=entry.get("answerable", True),
                relevant_docs=entry.get("relevant_docs", []),
                reference_answer=entry.get("reference_answer", ""),
            )
        )
    return items


def evaluate_question(
    pipeline: RagPipeline,
    client: LLMProvider,
    judge_model: str,
    item: EvalItem,
    *,
    retrieval_only: bool = False,
) -> QuestionResult:
    logger.debug("eval id=%s category=%s retrieval_only=%s", item.id, item.category, retrieval_only)
    start = time.perf_counter()
    if retrieval_only:
        hits = pipeline.retriever.retrieve(item.question)
        retrieval_seconds = time.perf_counter() - start
        total_seconds = retrieval_seconds
        answer = ""
        retrieved_docs = [h.chunk.doc_id for h in hits]
        context = ""
    else:
        result = pipeline.answer(item.question)
        retrieval_seconds = result.retrieval_seconds
        total_seconds = result.total_seconds
        answer = result.answer
        retrieved_docs = [h.chunk.doc_id for h in result.hits]
        context = "\n\n".join(h.chunk.text for h in result.hits)

    relevant = set(item.relevant_docs)

    scores: dict = {}
    if item.answerable and relevant:
        scores["hit_rate"] = metrics.hit_rate_at_k(retrieved_docs, relevant)
        scores["precision"] = metrics.precision_at_k(retrieved_docs, relevant)
        scores["recall"] = metrics.recall_at_k(retrieved_docs, relevant)
        scores["mrr"] = metrics.mrr(retrieved_docs, relevant)
        scores["ndcg"] = metrics.ndcg_at_k(retrieved_docs, relevant)

    if not retrieval_only:
        if item.answerable:
            scores["token_f1"] = metrics.token_f1(answer, item.reference_answer)
            scores["faithfulness"] = metrics.judge_faithfulness(client, judge_model, context, answer)
            scores["relevance"] = metrics.judge_relevance(client, judge_model, item.question, answer)
            scores["correctness"] = metrics.judge_correctness(
                client, judge_model, item.question, item.reference_answer, answer
            )
            scores["abstained"] = metrics.is_abstention(answer)
        else:
            abstained = metrics.is_abstention(answer)
            scores["abstention_correct"] = 1.0 if abstained else 0.0

    logger.debug(
        "eval id=%s retrieval_s=%.2f total_s=%.2f scores=%s",
        item.id,
        retrieval_seconds,
        total_seconds,
        {k: v for k, v in scores.items() if v is not None},
    )
    return QuestionResult(
        item=item,
        answer=answer,
        retrieved_docs=retrieved_docs,
        scores=scores,
        retrieval_seconds=retrieval_seconds,
        total_seconds=total_seconds,
    )


_AGG_KEYS = [
    "hit_rate", "precision", "recall", "mrr", "ndcg",
    "token_f1", "faithfulness", "relevance", "correctness", "abstention_correct",
]


def _aggregate(results: list[QuestionResult]) -> dict:
    agg: dict = {}
    for key in _AGG_KEYS:
        values = [r.scores[key] for r in results if r.scores.get(key) is not None]
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    latencies = [r.total_seconds for r in results]
    retrievals = [r.retrieval_seconds for r in results]
    if latencies:
        agg["latency_mean_s"] = round(statistics.mean(latencies), 2)
        agg["latency_p95_s"] = round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2)
        agg["retrieval_mean_s"] = round(statistics.mean(retrievals), 3)
    agg["n"] = len(results)
    return agg


def run_evaluation(
    config: RagConfig,
    client: LLMProvider,
    pipeline: RagPipeline,
    limit: int | None = None,
    category: str | None = None,
    progress=None,
    retrieval_only: bool = False,
) -> dict:
    items = load_eval_set(config.eval_path)
    if category:
        items = [i for i in items if i.category == category]
    if limit:
        items = items[:limit]
    if not items:
        raise ValueError("No eval questions match the given filters")

    judge_model = config.effective_judge_model()
    logger.info(
        "eval questions=%d judge=%s category=%s limit=%s retrieval_only=%s",
        len(items),
        judge_model,
        category or "all",
        limit or "none",
        retrieval_only,
    )
    results: list[QuestionResult] = []
    for item in items:
        result = evaluate_question(
            pipeline, client, judge_model, item, retrieval_only=retrieval_only
        )
        results.append(result)
        if progress:
            progress(result)

    categories = sorted({r.item.category for r in results})
    report = {
        "config": {
            "chat_model": config.chat_model,
            "embed_model": config.embed_model,
            "judge_model": judge_model,
            "top_k": config.top_k,
            "chunk_chars": config.chunk_chars,
            "hybrid_enabled": config.hybrid_enabled,
            "hybrid_alpha": config.hybrid_alpha,
            "max_chunks_per_doc": config.max_chunks_per_doc,
            "retrieval_only": retrieval_only,
            "fallback_enabled": False if retrieval_only else config.fallback_enabled,
        },
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": _aggregate(results),
        "by_category": {
            cat: _aggregate([r for r in results if r.item.category == cat]) for cat in categories
        },
        "questions": [
            {
                "id": r.item.id,
                "category": r.item.category,
                "question": r.item.question,
                "answerable": r.item.answerable,
                "answer": r.answer,
                "reference_answer": r.item.reference_answer,
                "relevant_docs": r.item.relevant_docs,
                "retrieved_docs": r.retrieved_docs,
                "scores": r.scores,
                "retrieval_seconds": round(r.retrieval_seconds, 3),
                "total_seconds": round(r.total_seconds, 2),
            }
            for r in results
        ],
    }
    config.eval_results_path.parent.mkdir(parents=True, exist_ok=True)
    config.eval_results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "eval done questions=%d path=%s hit_rate=%s",
        len(results),
        config.eval_results_path,
        report["overall"].get("hit_rate", "-"),
    )
    return report
