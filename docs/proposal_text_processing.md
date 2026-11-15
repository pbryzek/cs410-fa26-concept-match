# Proposal Text Processing

This project maintains a canonical proposal export and adds a separate processed text view for analysis work.

## Purpose

- `data/processed/proposals.jsonl` remains the primary exported proposal corpus from `scripts/export_proposals.py`.
- `scripts/preprocess_proposals.py` reads that export and adds normalized token fields for text-based analysis.
- The processed output provides a reusable tokenized text representation for analysis workflows.

## Output files

- `data/processed/proposals_preprocessed.jsonl`
  - one processed JSON object per proposal
- `data/processed/corpus_stats.json`
  - corpus-level frequency summaries and token statistics

## Processed record fields

Each record extends the canonical export record with the following fields:

- `tokens`: list of lowercase tokens after normalization and stopword filtering.
- `processed_text`: concatenated token string used for modeling and feature extraction.
- `processed_token_count`: integer count of tokens in the record.

## Example

```json
{
  "submission_id": "6adbb65da8567d79c2f75a5e",
  "title": "Tech Review: Agent Memory and Context",
  "description_clean": "Community: Presentation Video: aidanjs7 techreview fa2026",
  "clean_text": "Tech Review: Agent Memory and Context Community: Presentation Video: aidanjs7 techreview fa2026 #fa2026 #techreview",
  "tokens": ["tech", "review", "agent", "memory", "context", "community", "presentation", "video", "aidanjs7", "techreview", "fa2026"],
  "processed_text": "tech review agent memory context community presentation video aidanjs7 techreview fa2026",
  "processed_token_count": 11
}
```

## Notes

- The tokenizer lowercases text and keeps words and simple hyphenated forms.
