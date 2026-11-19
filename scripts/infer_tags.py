#!/usr/bin/env python3
"""Infer CS410 concept tags using trained tomotopy LDA model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.lda.model import LDAModel
from backend.app.lda.tagger import CS410ConceptTagger, TOKEN_RE


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer top-k CS410 concept tags using tomotopy.")
    parser.add_argument("--text", default=None, help="Raw text to tag")
    parser.add_argument("--id", default=None, help="Submission ID from preprocessed corpus")
    parser.add_argument("--top-k", type=int, default=5, help="Number of tags to return")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold to display")
    parser.add_argument("--model", default="data/processed/lda_model.bin", help="Path to trained tomotopy model")
    parser.add_argument("--corpus", default="data/processed/proposals_preprocessed.jsonl", help="Preprocessed corpus path")

    args = parser.parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        sys.exit(f"Trained model not found at {model_path}. Run scripts/train_lda.py first.")

    model = LDAModel.load(model_path)
    tagger = CS410ConceptTagger(model=model)

    if args.id:
        corpus_path = Path(args.corpus)
        found = None
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("submission_id") == args.id:
                    found = rec
                    break
        if not found:
            sys.exit(f"Proposal ID {args.id} not found in {corpus_path}")

        print(f"\nProposal: {found.get('title')}")
        print(f"Original Hashtags: {found.get('hashtags')}")
        tags = tagger.infer_tags(tokens=found.get("tokens", []), existing_tags=found.get("hashtags", []), top_k=args.top_k)
    else:
        text = args.text or "Rocchio"
        print(f"\nInput Text: \"{text}\"")
        tokens = TOKEN_RE.findall(text.lower())
        tags = tagger.infer_tags(tokens=tokens, top_k=args.top_k)

    print(f"\n{'Rank':>4} | {'Concept Tag':<25} | {'Score':>7} | Matched Keywords")
    print("-" * 65)
    rank = 1
    for t in tags:
        if t["score"] < args.min_score:
            continue
        kw_str = ", ".join(t["matched_keywords"]) if t["matched_keywords"] else "(topic inferred)"
        print(f"{rank:4d} | {t['tag']:<25} | {t['score']:7.2f} | {kw_str}")
        rank += 1


if __name__ == "__main__":
    main()
