"""TF-IDF vectors and cosine similarity over pre-tokenized documents."""

from collections import Counter
from typing import Dict, Iterable, List, Mapping
import math

Vector = Dict[str, float]


class TfidfModel:
    """IDF fitted on one corpus; other texts are projected into its vocabulary.

    Weights are (1 + ln tf) · (ln((N + 1) / (df + 1)) + 1), L2-normalised, so the
    dot product of two vectors is their cosine similarity. Terms the fitted
    corpus has never seen are dropped.
    """

    def __init__(self, docs: Mapping[str, List[str]]):
        self.num_docs = len(docs)
        doc_freq: Counter = Counter()
        for tokens in docs.values():
            doc_freq.update(set(tokens))
        self.idf: Dict[str, float] = {
            term: math.log((self.num_docs + 1) / (df + 1)) + 1.0 for term, df in doc_freq.items()
        }
        self.doc_vectors: Dict[str, Vector] = {
            doc_id: self.vectorize(tokens) for doc_id, tokens in docs.items()
        }

    def vectorize(self, tokens: Iterable[str]) -> Vector:
        counts = Counter(t for t in tokens if t in self.idf)
        vector = {term: (1.0 + math.log(tf)) * self.idf[term] for term, tf in counts.items()}
        norm = math.sqrt(sum(w * w for w in vector.values()))
        return {term: w / norm for term, w in vector.items()} if norm else {}


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two L2-normalised sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())
