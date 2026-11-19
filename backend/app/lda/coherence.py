"""Topic coherence evaluation using tomotopy's built-in Coherence module."""

from __future__ import annotations

from typing import Any, Dict, List

import tomotopy as tp


def compute_coherence(
    model: tp.LDAModel,
    metric: str = "u_mass",
    top_n: int = 10,
) -> float:
    """Compute overall topic coherence using tomotopy.coherence.Coherence.

    Supported metrics: 'u_mass', 'c_npmi', 'c_v', 'c_uci'.
    """
    coh = tp.coherence.Coherence(model, coherence=metric, top_n=top_n)
    return round(float(coh.get_score()), 4)


def evaluate_topic_counts(
    documents: List[List[str]],
    k_candidates: List[int],
    iterations: int = 60,
    burn_in: int = 20,
    min_df: int = 3,
    rm_top: int = 5,
    seed: int = 42,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Evaluate candidate topic counts (K) using tomotopy's built-in coherence."""
    results: List[Dict[str, Any]] = []

    for k in k_candidates:
        mdl = tp.LDAModel(
            k=k,
            min_df=min_df,
            rm_top=rm_top,
            seed=seed,
        )
        for doc in documents:
            mdl.add_doc(doc)

        mdl.burn_in = burn_in
        mdl.train(iterations, workers=1)

        u_mass = compute_coherence(mdl, metric="u_mass", top_n=top_n)
        c_npmi = compute_coherence(mdl, metric="c_npmi", top_n=top_n)

        results.append({
            "k": k,
            "c_umass": u_mass,
            "c_npmi": c_npmi,
            "perplexity": round(float(mdl.perplexity), 2),
        })

    return results
