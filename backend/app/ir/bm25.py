"""Minimal in-memory BM25 index.

This is a stand-in for the team's BM25 baseline so RM3 is not blocked on it.
RM3 only relies on the small surface below, so the real ranker can replace
this class as long as it provides:

    search(query, k) -> [(doc_id, score)]   query: term -> weight, or a list of terms
    doc_term_counts(doc_id) -> {term: tf}
    doc_length(doc_id) -> int
    doc_freq(term) -> int
    num_docs -> int
"""

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union
import math

Query = Union[Mapping[str, float], Iterable[str]]


def as_weighted_query(query: Query) -> Dict[str, float]:
    """Normalise a query to term -> weight. A term list becomes term counts."""
    if isinstance(query, Mapping):
        return {term: float(weight) for term, weight in query.items() if weight > 0}
    return {term: float(count) for term, count in Counter(query).items()}


class BM25Index:
    """Okapi BM25 over pre-tokenized documents, with weighted query terms."""

    def __init__(self, docs: Mapping[str, List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf: Dict[str, Counter] = {doc_id: Counter(tokens) for doc_id, tokens in docs.items()}
        self._len: Dict[str, int] = {doc_id: len(tokens) for doc_id, tokens in docs.items()}
        self._postings: Dict[str, Dict[str, int]] = {}
        for doc_id, counts in self._tf.items():
            for term, tf in counts.items():
                self._postings.setdefault(term, {})[doc_id] = tf
        self.num_docs = len(self._tf)
        self.avg_doc_length = (sum(self._len.values()) / self.num_docs) if self.num_docs else 0.0

    def doc_term_counts(self, doc_id: str) -> Mapping[str, int]:
        return self._tf[doc_id]

    def doc_length(self, doc_id: str) -> int:
        return self._len[doc_id]

    def doc_freq(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Lucene-style BM25 IDF; never negative, even for very common terms."""
        df = self.doc_freq(term)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: Query, k: Optional[int] = 10) -> List[Tuple[str, float]]:
        """Rank documents for a query. ``k=None`` returns every matching document."""
        scores: Dict[str, float] = {}
        for term, weight in as_weighted_query(query).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                norm = 1.0 - self.b + self.b * self._len[doc_id] / self.avg_doc_length
                term_score = idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * term_score
        # Ties break on doc_id so rankings are deterministic.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked if k is None else ranked[:k]
