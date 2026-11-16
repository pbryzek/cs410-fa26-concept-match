#!/usr/bin/env python3
"""Run a query through BM25 and BM25+RM3 and show both rankings side by side.

Usage:
    python3 scripts/rm3_search.py "relevance feedback"
    python3 scripts/rm3_search.py "topic model" --fb-docs 5 --fb-terms 15 --feedback-weight 0.6
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ir import BM25Index, RM3Config, load_corpus, load_stopwords, rm3_search, tokenize  # noqa: E402


def add_rm3_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = RM3Config()
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data/processed/proposals_preprocessed.jsonl"))
    parser.add_argument("--fb-docs", type=int, default=defaults.fb_docs)
    parser.add_argument("--fb-terms", type=int, default=defaults.fb_terms)
    parser.add_argument("--feedback-weight", type=float, default=defaults.feedback_weight)
    parser.add_argument("--min-df", type=int, default=defaults.min_df)
    parser.add_argument("--max-df-ratio", type=float, default=defaults.max_df_ratio)


def config_from_args(args: argparse.Namespace) -> RM3Config:
    return RM3Config(
        fb_docs=args.fb_docs,
        fb_terms=args.fb_terms,
        feedback_weight=args.feedback_weight,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and BM25+RM3 for one query.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=10, help="Results to show per system")
    add_rm3_arguments(parser)
    args = parser.parse_args()

    docs, records = load_corpus(args.corpus)
    index = BM25Index(docs)
    query_terms = tokenize(args.query, load_stopwords())
    if not query_terms:
        sys.exit("Query has no searchable terms after stopword removal.")

    baseline = index.search(query_terms, args.k)
    expanded = rm3_search(index, query_terms, args.k, config_from_args(args))
    baseline_ids = {doc_id for doc_id, _ in baseline}

    print("Query terms:     " + " ".join(query_terms))
    print("Expansion terms: " + ", ".join("%s (%.3f)" % pair for pair in expanded.expansion_terms))
    for name, ranking in (("BM25", baseline), ("BM25 + RM3", expanded.results)):
        print("\n" + name)
        for rank, (doc_id, score) in enumerate(ranking, start=1):
            marker = "+" if name != "BM25" and doc_id not in baseline_ids else " "
            print("%2d. %s %7.3f  %s" % (rank, marker, score, records[doc_id].get("title", "")[:90]))
    print("\n+ = not in the BM25 top %d" % args.k)


if __name__ == "__main__":
    main()
