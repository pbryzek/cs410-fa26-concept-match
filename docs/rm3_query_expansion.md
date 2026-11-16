# RM3 Query Expansion

RM3 adds related terms to a search query using the top results of a first-pass
retrieval (pseudo-relevance feedback), so a query like "relevance feedback" can
also reach proposals that only say "Rocchio".

## Where the code lives

| Path | Purpose |
| --- | --- |
| `backend/app/ir/rm3.py` | RM1 relevance model, RM3 interpolation, two-pass search |
| `backend/app/ir/bm25.py` | Minimal in-memory BM25, a stand-in until the team baseline lands |
| `backend/app/ir/tokenizer.py` | Query tokenizer, mirrors `scripts/preprocess_proposals.py` |
| `backend/app/ir/metrics.py` | Precision@K, Average Precision, nDCG@K |
| `scripts/rm3_search.py` | Compare BM25 and BM25+RM3 for one query |
| `scripts/evaluate_rm3.py` | Build a judging pool and score both systems |
| `backend/tests/test_ir.py` | Unit tests (no server, Mongo or OpenSearch needed) |

## How it works

1. Run the query through BM25 and keep the top `fb_docs` documents.
2. Estimate a relevance model (RM1) from them:
   `P(w|R) ∝ Σ_D (tf(w,D) / |D|) · score(D)`, with scores normalised over the feedback set.
3. Drop candidate terms that are too common (`max_df_ratio`) or too rare (`min_df`),
   keep the top `fb_terms`, and renormalise. Original query terms are never dropped.
4. Interpolate with the original query (RM3):
   `P'(w|Q) = (1 − λ) · P_mle(w|Q) + λ · P(w|R)`, where λ is `feedback_weight`.
5. Search again with the expanded query, using the probabilities as term weights.

`fb_terms` counts every term in the feedback model, including original query
terms that also rank there, so the number of *new* terms can be smaller.

Defaults: `fb_docs=10`, `fb_terms=10`, `feedback_weight=0.5`, `min_df=2`, `max_df_ratio=0.3`.

## Usage

Build the corpus first (the `data/` folder is gitignored):

```bash
python3 scripts/export_proposals.py
python3 scripts/preprocess_proposals.py
```

Try a query:

```bash
python3 scripts/rm3_search.py "relevance feedback"
python3 scripts/rm3_search.py "topic model" --fb-docs 5 --fb-terms 15 --feedback-weight 0.6
```

From Python:

```python
from app.ir import BM25Index, RM3Config, load_corpus, load_stopwords, rm3_search, tokenize

docs, records = load_corpus("data/processed/proposals_preprocessed.jsonl")
index = BM25Index(docs)
result = rm3_search(index, tokenize("relevance feedback", load_stopwords()), k=10)
result.results          # [(submission_id, score), ...]
result.expansion_terms  # [(term, weight), ...] new terms only, for display in the UI
```

## Evaluation

```bash
# 1. Pool the top results of both systems for labeling
python3 scripts/evaluate_rm3.py pool --queries data/eval/queries.tsv --output data/eval/pool.tsv

# 2. Fill in the relevance column (0 = not relevant, 1 = relevant, 2 = highly relevant), then:
python3 scripts/evaluate_rm3.py eval --queries data/eval/queries.tsv --qrels data/eval/pool.tsv --per-query
```

`queries.tsv` is tab-separated with a header row: `query_id`, `query`.

Label the pool, not just BM25's results. Unjudged documents score as
non-relevant, so judging only the baseline's top results would penalise every
new document RM3 finds. The `unjudged@K` column in the report shows how many
top-K results per system had no label.

## Swapping in the team BM25

`rm3.py` only calls five things on the index: `search(query, k)`,
`doc_term_counts(doc_id)`, `doc_length(doc_id)`, `doc_freq(term)` and `num_docs`.
`search` must accept a `{term: weight}` mapping and return `[(doc_id, score)]`
with non-negative scores. Any ranker with that surface can replace `BM25Index`.

## Tests

```bash
pip install pytest
pytest backend/tests/test_ir.py
```
