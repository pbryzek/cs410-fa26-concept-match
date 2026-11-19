#!/usr/bin/env python3
"""Train an LDA topic model on CS410 proposals using tomotopy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.lda.coherence import compute_coherence, evaluate_topic_counts
from backend.app.lda.model import LDAModel
from backend.app.lda.tagger import CS410ConceptTagger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tomotopy LDA and infer CS410 concept tags.")
    parser.add_argument("--input", default="data/processed/proposals_preprocessed.jsonl", help="Input preprocessed JSONL")
    parser.add_argument("--output-model", default="data/processed/lda_model.bin", help="Output path for trained tomotopy model")
    parser.add_argument("--output-tagged", default="data/processed/proposals_tagged.jsonl", help="Output path for tagged proposals JSONL")
    parser.add_argument("--k", type=int, default=10, help="Number of topics K")
    parser.add_argument("--evaluate-k", action="store_true", help="Evaluate candidate topic counts using tomotopy coherence")
    parser.add_argument("--k-candidates", default="5,8,10,12,15", help="Comma-separated list of candidate K values")
    parser.add_argument("--iterations", type=int, default=80, help="Gibbs sampling iterations")
    parser.add_argument("--burn-in", type=int, default=25, help="Burn-in iterations")
    parser.add_argument("--min-df", type=int, default=3, help="Minimum document frequency")
    parser.add_argument("--rm-top", type=int, default=5, help="Number of top words to prune")
    parser.add_argument("--top-k-tags", type=int, default=5, help="Number of concept tags to infer per proposal")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # Load preprocessed proposals
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    docs_tokens = [r.get("tokens", []) for r in records]
    print(f"Loaded {len(records)} proposal records.")

    # 1. (Optional) Evaluate topic counts using built-in coherence
    selected_k = args.k
    if args.evaluate_k:
        k_list = [int(x.strip()) for x in args.k_candidates.split(",") if x.strip()]
        print(f"\nEvaluating topic counts {k_list} using tomotopy coherence...")
        results = evaluate_topic_counts(
            documents=docs_tokens,
            k_candidates=k_list,
            iterations=args.iterations,
            burn_in=args.burn_in,
            min_df=args.min_df,
            rm_top=args.rm_top,
            seed=args.seed,
        )
        print(f"\n{'K':>4} | {'C_UMass':>9} | {'C_NPMI':>8} | {'Perplexity':>10}")
        print("-" * 40)
        best_k = k_list[0]
        best_npmi = -999.0
        for res in results:
            print(f"{res['k']:4d} | {res['c_umass']:9.4f} | {res['c_npmi']:8.4f} | {res['perplexity']:10.1f}")
            if res["c_npmi"] > best_npmi:
                best_npmi = res["c_npmi"]
                best_k = res["k"]
        print(f"\nRecommended optimal K: {best_k}")
        selected_k = best_k

    # 2. Train tomotopy model
    print(f"\nTraining tomotopy LDA model with K = {selected_k}...")
    model = LDAModel(
        k=selected_k,
        min_df=args.min_df,
        rm_top=args.rm_top,
        seed=args.seed,
    )
    model.add_documents(docs_tokens)
    model.train(iterations=args.iterations, burn_in=args.burn_in)

    u_mass = compute_coherence(model.model, metric="u_mass")
    c_npmi = compute_coherence(model.model, metric="c_npmi")
    print(f"Trained model: K={selected_k}, Vocab={model.num_vocabs}, Perplexity={model.perplexity:.1f}")
    print(f"Coherence: C_UMass={u_mass}, C_NPMI={c_npmi}")

    print("\nDiscovered Topics:")
    for k in range(selected_k):
        top_w = model.get_topic_words(k, top_n=8)
        w_str = ", ".join(f"{w} ({p:.3f})" for w, p in top_w)
        print(f"  Topic {k:2d}: {w_str}")

    # 3. Save model
    model_path = Path(args.output_model)
    model.save(model_path)
    print(f"\nSaved tomotopy model to {model_path}")

    # 4. Infer tags for dataset proposals
    print(f"Inferring top-{args.top_k_tags} CS410 concept tags for dataset...")
    tagger = CS410ConceptTagger(model=model)
    tagged_records = []
    for r in records:
        tagged_records.append(tagger.tag_proposal_record(r, top_k=args.top_k_tags))

    tagged_path = Path(args.output_tagged)
    tagged_path.parent.mkdir(parents=True, exist_ok=True)
    with tagged_path.open("w", encoding="utf-8") as f:
        for rec in tagged_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(tagged_records)} tagged proposals to {tagged_path}")


if __name__ == "__main__":
    main()
