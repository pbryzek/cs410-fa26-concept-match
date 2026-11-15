#!/usr/bin/env python3
"""Create a tokenized text view for the proposal corpus.

This script reads the canonical proposal export and adds normalized token fields
for downstream analysis and modeling workflows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def load_stopwords(stopwords_path: Path) -> set[str]:
    """Load a lowercase stopword set from the repo's shared stopwords file."""
    stopwords: set[str] = set()
    if stopwords_path.exists():
        for line in stopwords_path.read_text(encoding="utf-8").splitlines():
            token = line.strip().lower()
            if token:
                stopwords.add(token)
    return stopwords


def normalize_tokens(text: str, stopwords: set[str]) -> list[str]:
    """Lowercase, tokenize, and strip common stopwords from a text string."""
    lower = (text or "").lower()
    tokens = []
    for token in TOKEN_RE.findall(lower):
        if token in stopwords:
            continue
        if token in {"-", "--"}:
            continue
        tokens.append(token)
    return tokens


def build_processed_record(record: dict, stopwords: set[str]) -> dict:
    """Create a processed representation for one proposal record."""
    text = (
        record.get("clean_text")
        or " ".join(
            part for part in [record.get("title", ""), record.get("description_clean", "")] if part
        )
    )
    tokens = normalize_tokens(text, stopwords)
    processed = dict(record)
    processed["tokens"] = tokens
    processed["processed_text"] = " ".join(tokens)
    processed["processed_token_count"] = len(tokens)
    return processed


def compute_corpus_stats(records: list[dict]) -> dict:
    """Compute simple corpus-level statistics for the processed proposal records."""
    token_counts = [len(record.get("tokens", [])) for record in records]
    total_tokens = sum(token_counts)
    all_tokens = [token for record in records for token in record.get("tokens", [])]
    token_counts_map = Counter(all_tokens)

    stats = {
        "total_records": len(records),
        "total_tokens": total_tokens,
        "avg_tokens_per_record": round(total_tokens / len(records), 2) if records else 0.0,
        "unique_tokens": len(token_counts_map),
        "top_terms": [
            {"term": term, "count": count}
            for term, count in token_counts_map.most_common(25)
        ],
        "rows_with_tokens": sum(1 for count in token_counts if count > 0),
    }
    return stats


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from disk into Python dictionaries."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write processed records back to disk as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Create a processed-text layer for proposal data.")
    parser.add_argument("--input", default="data/processed/proposals.jsonl", help="Input JSONL export file")
    parser.add_argument("--output", default="data/processed/proposals_preprocessed.jsonl", help="Processed JSONL output")
    parser.add_argument(
        "--stats-output",
        default="data/processed/corpus_stats.json",
        help="Corpus-level statistics output JSON",
    )
    parser.add_argument(
        "--stopwords",
        default="backend/stopwords.txt",
        help="Stopword file used for token filtering",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)
    stopwords_path = Path(args.stopwords)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = read_jsonl(input_path)
    stopwords = load_stopwords(stopwords_path)
    processed_records = [build_processed_record(record, stopwords) for record in records]
    write_jsonl(output_path, processed_records)

    stats = compute_corpus_stats(processed_records)
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Processed {len(processed_records)} records to {output_path}")
    print(json.dumps({
        "total_records": stats["total_records"],
        "total_tokens": stats["total_tokens"],
        "avg_tokens_per_record": stats["avg_tokens_per_record"],
        "unique_tokens": stats["unique_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
