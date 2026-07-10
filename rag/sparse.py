"""Lightweight BM25 for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.doc_freq: Counter[str] = Counter()
        for toks in self.doc_tokens:
            self.doc_freq.update(set(toks))
        self.n_docs = len(self.doc_tokens)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return [0.0] * self.n_docs
        q_tf = Counter(q_tokens)
        out = [0.0] * self.n_docs
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            dl = self.doc_len[i]
            for term, q_weight in q_tf.items():
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += q_weight * self._idf(term) * (freq * (self.k1 + 1.0)) / denom
            out[i] = score
        return out
