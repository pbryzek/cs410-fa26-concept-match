#!/usr/bin/env python3
"""Evaluate BM25 against BM25+RM3 on a labeled query set.

Two steps:

1. Build a judging pool (the union of both systems' top results), so RM3's new
   documents get labeled instead of being counted as non-relevant:

       python3 scripts/evaluate_rm3.py pool --queries data/eval/queries.tsv --output data/eval/pool.tsv

2. After the team fills in the ``relevance`` column, score both systems:

       python3 scripts/evaluate_rm3.py eval --queries data/eval/queries.tsv --qrels data/eval/pool.tsv

File formats (tab-separated, with a header row):
    queries:  query_id, query
    qrels:    query_id, doc_id, relevance   (0 = not relevant; extra columns are ignored)
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ir import BM25Index, load_corpus, load_stopwords, rm3_search, tokenize  # noqa: E402
from app.ir.metrics import average_precision, ndcg_at_k, precision_at_k  # noqa: E402
from rm3_search import add_rm3_arguments, config_from_args  # noqa: E402


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_qrels(path):
    """Return ({query_id: {doc_id: grade}}, number of rows left unlabeled)."""
    qrels, unlabeled = {}, 0
    for row in read_tsv(path):
        grade = (row.get("relevance") or "").strip()
        if not grade:
            unlabeled += 1
            continue
        qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(grade)
    return qrels, unlabeled


def run_systems(index, query_terms, config):
    """Full rankings (doc ids, best first) for each system."""
    return {
        "BM25": [doc_id for doc_id, _ in index.search(query_terms, None)],
        "BM25+RM3": [doc_id for doc_id, _ in rm3_search(index, query_terms, None, config).results],
    }


def command_pool(args, index, records, stopwords):
    rows = []
    for query in read_tsv(args.queries):
        runs = run_systems(index, tokenize(query["query"], stopwords), config_from_args(args))
        pooled = sorted({doc_id for ranking in runs.values() for doc_id in ranking[: args.depth]})
        rows.extend((query["query_id"], doc_id, "", query["query"], records[doc_id].get("title", ""))
                    for doc_id in pooled)
    output = Path(args.output)
    if output.exists() and not args.force:
        sys.exit("%s already exists; pass --force to overwrite it (this discards any labels)." % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance", "query", "title"])
        writer.writerows(rows)
    print("Wrote %d (query, document) pairs to %s. Fill in the relevance column." % (len(rows), output))


def command_eval(args, index, records, stopwords):
    qrels, unlabeled = read_qrels(args.qrels)
    if unlabeled:
        print("Warning: %d rows in %s have no relevance label and were skipped." % (unlabeled, args.qrels))
    metric_names = ["P@%d" % args.k, "MAP", "nDCG@%d" % args.k, "unjudged@%d" % args.k]
    totals, evaluated = {}, 0
    for query in read_tsv(args.queries):
        judgments = qrels.get(query["query_id"])
        if not judgments:
            print("Skipping query %s: no judgments." % query["query_id"])
            continue
        evaluated += 1
        runs = run_systems(index, tokenize(query["query"], stopwords), config_from_args(args))
        for system, ranking in runs.items():
            values = [
                precision_at_k(ranking, judgments, args.k),
                average_precision(ranking, judgments),
                ndcg_at_k(ranking, judgments, args.k),
                float(sum(1 for doc_id in ranking[: args.k] if doc_id not in judgments)),
            ]
            if args.per_query:
                print("%-10s %-9s %s" % (query["query_id"], system, "  ".join("%.4f" % v for v in values)))
            sums = totals.setdefault(system, [0.0] * len(values))
            for i, value in enumerate(values):
                sums[i] += value
    if not evaluated:
        sys.exit("No queries with judgments; nothing to evaluate.")
    print("\nMean over %d queries" % evaluated)
    print("%-9s " % "system" + "  ".join("%12s" % name for name in metric_names))
    for system, sums in totals.items():
        print("%-9s " % system + "  ".join("%12.4f" % (value / evaluated) for value in sums))
    print("\nunjudged@%d is the mean number of top-%d results with no label; they score as "
          "non-relevant, so a high value means that system is being under-credited." % (args.k, args.k))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool = subparsers.add_parser("pool", help="Write the union of both systems' top results for labeling")
    pool.add_argument("--queries", required=True)
    pool.add_argument("--output", required=True)
    pool.add_argument("--depth", type=int, default=10, help="Top results taken from each system")
    pool.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    add_rm3_arguments(pool)

    evaluate = subparsers.add_parser("eval", help="Score BM25 and BM25+RM3 against judgments")
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--qrels", required=True)
    evaluate.add_argument("-k", type=int, default=10)
    evaluate.add_argument("--per-query", action="store_true")
    add_rm3_arguments(evaluate)

    args = parser.parse_args()
    docs, records = load_corpus(args.corpus)
    handler = command_pool if args.command == "pool" else command_eval
    handler(args, BM25Index(docs), records, load_stopwords())


if __name__ == "__main__":
    main()
