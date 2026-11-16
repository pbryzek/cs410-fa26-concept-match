"""RM3 query expansion (pseudo-relevance feedback).

RM1 (Lavrenko & Croft, 2001) estimates a relevance model from the top-ranked
documents of a first-pass retrieval:

    P(w|R)  ∝  Σ_{D in top-k}  P(w|D) · P(Q|D)

RM3 (Abdul-Jaleel et al., 2004) interpolates it with the original query so the
expanded query does not drift away from what the user typed:

    P'(w|Q) = (1 - λ) · P_mle(w|Q) + λ · P(w|R)

Because the first pass here is BM25 rather than query likelihood, P(Q|D) is
approximated by the document's first-pass score normalised over the feedback
set, which is the usual BM25+RM3 formulation.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from app.ir.bm25 import as_weighted_query


@dataclass(frozen=True)
class RM3Config:
    fb_docs: int = 10
    fb_terms: int = 10
    feedback_weight: float = 0.5
    min_df: int = 2
    max_df_ratio: float = 0.3
    exclude_terms: FrozenSet[str] = frozenset()


@dataclass
class RM3Result:
    results: List[Tuple[str, float]]                # final ranking: (doc_id, score)
    expanded_query: Dict[str, float]                # term -> weight used for the second pass
    expansion_terms: List[Tuple[str, float]]        # new terms only, best first
    feedback_docs: List[str] = field(default_factory=list)


def _query_model(query_terms: Iterable[str]) -> Dict[str, float]:
    counts = Counter(query_terms)
    total = sum(counts.values())
    return {term: count / total for term, count in counts.items()} if total else {}


def relevance_model(index, feedback: List[Tuple[str, float]]) -> Dict[str, float]:
    """RM1: P(w|R) over every term in the feedback documents (sums to 1)."""
    total_score = sum(score for _, score in feedback)
    if total_score <= 0:
        return {}
    model: Dict[str, float] = {}
    for doc_id, score in feedback:
        length = index.doc_length(doc_id)
        if not length:
            continue
        doc_weight = score / total_score
        for term, tf in index.doc_term_counts(doc_id).items():
            model[term] = model.get(term, 0.0) + doc_weight * tf / length
    return model


def expand_query(index, query_terms: List[str], config: Optional[RM3Config] = None) -> RM3Result:
    """Build the RM3 query model. ``results`` holds the first-pass ranking."""
    config = config or RM3Config()
    original = _query_model(query_terms)
    feedback = index.search(original, config.fb_docs) if original else []
    model = relevance_model(index, feedback)
    if not model or config.feedback_weight <= 0:
        return RM3Result(feedback, dict(original), [], [doc_id for doc_id, _ in feedback])

    max_df = config.max_df_ratio * index.num_docs

    def keep(term: str) -> bool:
        if term in original:
            return True
        if term in config.exclude_terms:
            return False
        return config.min_df <= index.doc_freq(term) <= max_df

    candidates = sorted(
        ((term, p) for term, p in model.items() if keep(term)),
        key=lambda item: (-item[1], item[0]),
    )[: config.fb_terms]
    mass = sum(p for _, p in candidates)
    feedback_model = {term: p / mass for term, p in candidates} if mass else {}

    lam = min(config.feedback_weight, 1.0) if feedback_model else 0.0
    expanded: Dict[str, float] = {}
    for term in set(original) | set(feedback_model):
        weight = (1.0 - lam) * original.get(term, 0.0) + lam * feedback_model.get(term, 0.0)
        if weight > 0:
            expanded[term] = weight

    new_terms = sorted(
        ((term, weight) for term, weight in expanded.items() if term not in original),
        key=lambda item: (-item[1], item[0]),
    )
    return RM3Result(feedback, expanded, new_terms, [doc_id for doc_id, _ in feedback])


def rm3_search(index, query_terms: List[str], k: Optional[int] = 10,
               config: Optional[RM3Config] = None) -> RM3Result:
    """First-pass retrieval, RM3 expansion, then a second pass with the expanded query."""
    expansion = expand_query(index, query_terms, config)
    if not expansion.expanded_query:
        return RM3Result([], {}, [], [])
    expansion.results = index.search(as_weighted_query(expansion.expanded_query), k)
    return expansion
