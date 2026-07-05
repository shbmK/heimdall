"""Evaluation metrics.

Retrieval metrics are computed at document level against labeled relevant
docs. Generation quality uses a mix of deterministic overlap (token F1) and
LLM-as-judge scores (faithfulness, relevance, correctness) on a 1-5 scale.
"""

from __future__ import annotations

import math
import re

from .llm import LLMError, OllamaClient

# --------------------------------------------------------------- retrieval


def hit_rate_at_k(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    return 1.0 if any(d in relevant_docs for d in retrieved_docs) else 0.0


def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    if not retrieved_docs:
        return 0.0
    return sum(1 for d in retrieved_docs if d in relevant_docs) / len(retrieved_docs)


def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    if not relevant_docs:
        return 0.0
    found = {d for d in retrieved_docs if d in relevant_docs}
    return len(found) / len(relevant_docs)


def mrr(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    """Binary-relevance nDCG over the retrieved ranking."""
    dcg = sum(
        (1.0 if doc in relevant_docs else 0.0) / math.log2(i + 1)
        for i, doc in enumerate(retrieved_docs, start=1)
    )
    ideal_hits = min(len(relevant_docs), len(retrieved_docs))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------- generation

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "and", "or",
    "to", "his", "her", "their", "its", "he", "she", "they", "it", "as", "by",
    "with", "for", "that", "this", "at", "from", "be", "been", "has", "have", "had",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return 0.0
    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    pred_counts: dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


_ABSTAIN_RE = re.compile(
    r"don'?t know|do not know|not in my knowledge|no information|not mentioned|"
    r"not (?:contained|available|found|present) in|cannot (?:find|answer)|unable to (?:find|answer)",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(answer))


# ------------------------------------------------------------ LLM-as-judge

_JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Reply with ONLY a single integer from 1 to 5. "
    "No words, no punctuation, just the number."
)

_FAITHFULNESS_PROMPT = """Rate how faithful the answer is to the context (is every claim in the answer supported by the context?).
1 = mostly fabricated, 3 = partially supported, 5 = fully supported by the context.

Context:
{context}

Answer:
{answer}

Rating (1-5):"""

_RELEVANCE_PROMPT = """Rate how well the answer addresses the question (regardless of whether it is factually correct).
1 = completely off-topic, 3 = partially addresses it, 5 = directly and completely addresses it.

Question:
{question}

Answer:
{answer}

Rating (1-5):"""

_CORRECTNESS_PROMPT = """Rate how well the answer agrees with the reference answer.
1 = contradicts or misses the reference entirely, 3 = partially matches, 5 = fully consistent with the reference.

Question:
{question}

Reference answer:
{reference}

Answer to evaluate:
{answer}

Rating (1-5):"""


def _judge(client: OllamaClient, model: str, prompt: str) -> float | None:
    """Run a 1-5 judge; returns None if the judge output is unusable."""
    try:
        raw = client.generate(prompt, system=_JUDGE_SYSTEM, model=model, temperature=0.0)
    except LLMError:
        return None
    m = re.search(r"[1-5]", raw)
    return float(m.group(0)) if m else None


def judge_faithfulness(client: OllamaClient, model: str, context: str, answer: str) -> float | None:
    return _judge(client, model, _FAITHFULNESS_PROMPT.format(context=context[:8000], answer=answer))


def judge_relevance(client: OllamaClient, model: str, question: str, answer: str) -> float | None:
    return _judge(client, model, _RELEVANCE_PROMPT.format(question=question, answer=answer))


def judge_correctness(client: OllamaClient, model: str, question: str, reference: str, answer: str) -> float | None:
    return _judge(client, model, _CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer))
