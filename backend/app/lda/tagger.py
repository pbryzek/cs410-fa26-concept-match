"""Automatic inference of CS410 concept tags using a trained tomotopy LDA model."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set

from backend.app.lda.model import LDAModel

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def match_anchor(anchor: str, token_set: Set[str]) -> bool:
    """Check if an anchor keyword matches token_set, handling hyphens, spaces, and concatenated forms."""
    if anchor in token_set:
        return True
    if "-" in anchor:
        # Check concatenated form e.g. "tfidf", "kmeans", "ngram"
        nodash = anchor.replace("-", "")
        if nodash in token_set:
            return True
        # Check all constituent parts e.g. {"tf", "idf"} for "tf-idf"
        parts = [p for p in anchor.split("-") if p]
        if parts and all(p in token_set for p in parts):
            return True
    return False


# Canonical CS410 Concept Taxonomy with anchor keywords
CS410_CONCEPT_TAXONOMY: Dict[str, Set[str]] = {
    "relevance-feedback": {
        "relevance",
        "feedback",
        "rocchio",
        "prf",
        "pseudo-relevance",
        "judgment",
        "reformulation",
        "formulation",
        "positive",
        "negative",
    },
    "bm25": {
        "bm25",
        "okapi",
        "ranking",
        "tf-idf",
        "term-frequency",
        "document-frequency",
        "saturation",
        "length-normalization",
    },
    "query-expansion": {
        "query",
        "expansion",
        "reformulation",
        "formulation",
        "rm3",
        "prf",
        "synonyms",
        "pseudo-relevance",
        "feedback",
        "expanded",
    },
    "dense-retrieval": {
        "dense",
        "retrieval",
        "embeddings",
        "embedding",
        "bi-encoder",
        "cross-encoder",
        "vector",
        "semantic",
        "faiss",
    },
    "semantic-search": {
        "semantic",
        "search",
        "embeddings",
        "similarity",
        "vector",
        "latent",
        "meaning",
        "dense",
    },
    "topic-modeling": {
        "topic",
        "topics",
        "modeling",
        "lda",
        "dirichlet",
        "gibbs",
        "coherence",
        "latent",
        "perplexity",
    },
    "text-classification": {
        "classification",
        "classifier",
        "categorization",
        "supervised",
        "labeling",
        "multiclass",
        "svm",
        "naive-bayes",
    },
    "text-clustering": {
        "clustering",
        "cluster",
        "k-means",
        "hierarchical",
        "unsupervised",
        "grouping",
    },
    "sentiment-analysis": {
        "sentiment",
        "opinion",
        "polarity",
        "positive",
        "negative",
        "reviews",
    },
    "rag": {
        "rag",
        "retrieval-augmented",
        "generation",
        "llm",
        "hallucination",
        "context",
        "prompting",
    },
    "evaluation-metrics": {
        "evaluation",
        "metrics",
        "precision",
        "recall",
        "ndcg",
        "map",
        "mrr",
        "f1",
        "benchmark",
    },
    "knowledge-graph": {
        "knowledge-graph",
        "entities",
        "relations",
        "triples",
        "entity-linking",
        "graph",
    },
    "language-models": {
        "language-models",
        "language-model",
        "unigram",
        "n-gram",
        "smoothing",
        "perplexity",
        "transformer",
        "bert",
    },
    "text-mining": {
        "text-mining",
        "pattern",
        "extraction",
        "nlp",
        "corpus",
        "mining",
        "frequency",
    },
    "question-answering": {
        "question",
        "answering",
        "qa",
        "passage",
        "reading-comprehension",
        "answer",
    },
    "recommendation-systems": {
        "recommendation",
        "recommender",
        "collaborative",
        "filtering",
        "items",
        "preference",
    },
    # Domains from the proposals
    "healthcare": {"healthcare", "health", "care", "clinical", "medical", "patient", "disease", "diagnosis", "hospital"},
    "finance": {"finance", "financial", "stock", "market", "earnings", "investment", "trading", "banking"},
    "cybersecurity": {"cybersecurity", "security", "vulnerability", "malware", "threat", "attack", "cyber"},
    "education": {"student", "course", "education", "learning", "exam", "quiz", "academic"},
    "resume-matching": {"resume", "job", "candidate", "skills", "applicant", "career", "hiring"},
    "code-search": {"code", "repository", "github", "programming", "software", "developer"},
}


class CS410ConceptTagger:
    """Infers top-k CS410 concept tags using a trained tomotopy LDA model."""

    def __init__(
        self,
        model: LDAModel,
        taxonomy: Optional[Dict[str, Set[str]]] = None,
        topic_weight: float = 0.65,
    ) -> None:
        self.model = model
        raw_tax = taxonomy or CS410_CONCEPT_TAXONOMY
        # Ensure concept name is always in its anchors
        self.taxonomy = {k: set(v) for k, v in raw_tax.items()}
        for concept in self.taxonomy:
            self.taxonomy[concept].add(concept.lower())

        self.topic_weight = topic_weight

        # Precompute topic-concept affinities from model's topic words
        self.topic_concept_affinity: List[Dict[str, float]] = []
        self._build_affinities()

    def _build_affinities(self) -> None:
        """Align each LDA topic with CS410 concepts based on top words."""
        self.topic_concept_affinity = []
        K = self.model.k

        for k in range(K):
            topic_words = dict(self.model.get_topic_words(k, top_n=80))
            concept_scores: Dict[str, float] = {}

            for concept, anchors in self.taxonomy.items():
                matched = [
                    w for w in anchors
                    if w in topic_words or ("-" in w and w.replace("-", "") in topic_words)
                ]
                if not matched:
                    concept_scores[concept] = 0.0
                else:
                    prob_sum = sum(topic_words.get(w, topic_words.get(w.replace("-", ""), 0.0)) for w in matched)
                    concept_scores[concept] = prob_sum
            self.topic_concept_affinity.append(concept_scores)

    def infer_tags(
        self,
        tokens: List[str],
        doc_topic_dist: Optional[List[float]] = None,
        existing_tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Infer top-k concept tags for a given token list."""
        if doc_topic_dist is None:
            doc_topic_dist = self.model.infer_topics(tokens)

        token_set = {t.lower() for t in tokens}
        existing_set = {t.lower().lstrip("#") for t in (existing_tags or [])}
        K = self.model.k

        concept_scores = {}
        for concept, anchors in self.taxonomy.items():
            topic_score = sum(
                doc_topic_dist[k] * self.topic_concept_affinity[k].get(concept, 0.0)
                for k in range(K)
            )

            matched_keywords = sorted([a for a in anchors if match_anchor(a, token_set)])
            if matched_keywords:
                lexical_score = 1.0 + 0.5 * (len(matched_keywords) - 1)
            else:
                lexical_score = 0.0

            tag_slug = concept.lower()
            tag_slug_nodash = tag_slug.replace("-", "")
            manual_boost = 1.5 if (tag_slug in existing_set or tag_slug_nodash in existing_set) else 0.0

            final_score = (
                lexical_score
                + self.topic_weight * topic_score
                + manual_boost
            )

            concept_scores[concept] = {
                "tag": concept,
                "score": round(final_score, 4),
                "matched_keywords": matched_keywords,
            }

        ranked = sorted(concept_scores.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def tag_proposal_record(self, record: dict, top_k: int = 5) -> dict:
        """Enrich a proposal record with inferred concept tags."""
        tokens = record.get("tokens", [])
        existing_tags = record.get("hashtags", [])

        inferred = self.infer_tags(tokens=tokens, existing_tags=existing_tags, top_k=top_k)

        enriched = dict(record)
        enriched["inferred_concept_tags"] = [item["tag"] for item in inferred]
        enriched["tag_details"] = inferred
        return enriched
