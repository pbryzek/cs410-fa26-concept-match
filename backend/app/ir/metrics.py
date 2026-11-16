"""Ranked-retrieval metrics: Precision@K, Average Precision, nDCG@K.

``ranking`` is a list of doc ids, best first. ``judgments`` maps doc id to a
relevance grade (0 = not relevant). Unjudged documents count as not relevant.
"""

from typing import List, Mapping
import math


def precision_at_k(ranking: List[str], judgments: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranking[:k] if judgments.get(doc_id, 0) > 0)
    return hits / k


def average_precision(ranking: List[str], judgments: Mapping[str, int]) -> float:
    """AP over the full ranking, normalised by the number of relevant documents."""
    num_relevant = sum(1 for grade in judgments.values() if grade > 0)
    if not num_relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        if judgments.get(doc_id, 0) > 0:
            hits += 1
            total += hits / rank
    return total / num_relevant


def _dcg(grades: List[int]) -> float:
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranking: List[str], judgments: Mapping[str, int], k: int) -> float:
    ideal = _dcg(sorted((g for g in judgments.values() if g > 0), reverse=True)[:k])
    if not ideal:
        return 0.0
    return _dcg([judgments.get(doc_id, 0) for doc_id in ranking[:k]]) / ideal
