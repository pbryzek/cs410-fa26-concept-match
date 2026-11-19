"""LDA topic modeling using the tomotopy library (C++ Collapsed Gibbs Sampling)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import tomotopy as tp


class LDAModel:
    """Clean wrapper around tomotopy.LDAModel."""

    def __init__(
        self,
        k: int = 10,
        alpha: Optional[float] = None,
        eta: float = 0.01,
        min_df: int = 3,
        rm_top: int = 5,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.alpha = (50.0 / k) if alpha is None else alpha
        self.eta = eta
        self.min_df = min_df
        self.rm_top = rm_top
        self.seed = seed

        self.model = tp.LDAModel(
            k=self.k,
            alpha=self.alpha,
            eta=self.eta,
            min_df=self.min_df,
            rm_top=self.rm_top,
            seed=self.seed,
        )

    def add_documents(self, documents: List[List[str]]) -> None:
        """Add tokenized documents to the model."""
        for doc in documents:
            self.model.add_doc(doc)

    def train(self, iterations: int = 80, burn_in: int = 25, workers: int = 1) -> None:
        """Train the model using Collapsed Gibbs Sampling."""
        self.model.burn_in = burn_in
        self.model.train(iterations, workers=workers)

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return the top words and probabilities for a topic."""
        return [(word, float(prob)) for word, prob in self.model.get_topic_words(topic_id, top_n=top_n)]

    def infer_topics(self, tokens: List[str]) -> List[float]:
        """Infer the topic distribution for new/unseen text tokens."""
        if not tokens:
            return [1.0 / self.k] * self.k
        try:
            doc = self.model.make_doc(tokens)
            if doc is None:
                return [1.0 / self.k] * self.k
            topic_dist, _ = self.model.infer(doc)
            return [float(p) for p in topic_dist]
        except Exception:
            return [1.0 / self.k] * self.k

    @property
    def perplexity(self) -> float:
        """Model perplexity on training data."""
        return float(self.model.perplexity)

    @property
    def num_vocabs(self) -> int:
        """Total vocabulary size after frequency filtering."""
        return int(self.model.num_vocabs)

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))

    @classmethod
    def load(cls, path: Path | str) -> "LDAModel":
        """Load a trained model from disk."""
        inst = cls.__new__(cls)
        inst.model = tp.LDAModel.load(str(path))
        inst.k = inst.model.k
        return inst
