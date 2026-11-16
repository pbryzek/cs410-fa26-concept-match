# Plan: RM3 Query Expansion, Resource Linking, and the Core MR

RM3 query expansion, the resource-linking engine, and managing the merge request back to the main repository.

## Findings that shaped the plan

Measured on `data/cs410_snapshot.json` (405 proposals):

- **Small, template-heavy corpus.** Median proposal is about 200 words. `#project`
  appears in 325 proposals, and template text ("Community:", "Presentation Video:",
  netIDs) is everywhere. Unfiltered RM3 would pick boilerplate as expansion terms.
- **Few real resource links.** 79 of 405 proposals contain an external URL, but there
  are only 27 unique URLs, and two of them are form-template placeholders
  (`https://github.com/...` in 61 proposals, `https://mediaspace.illinois.edu/...` in 60).
  That leaves 25 real resources: mostly GitHub repositories and Hugging Face datasets,
  with 2 YouTube videos.
- **No stemming yet.** `scripts/preprocess_proposals.py` lowercases and removes stop
  words only. Query and index tokenization must match, so this needs a team decision.
- **No BM25 baseline in the repo.** Live search is an OpenSearch `multi_match`
  (`backend/elastic/manage_data.py`). RM3 uses a stand-in BM25 behind a small interface.
- **`data/` is gitignored and existing tests need a live server and Mongo.** New tests
  use in-file fixtures and run without services.

## 1. RM3 query expansion (branch `rm3`) — implemented

Pure-Python module in `backend/app/ir/`; see `docs/rm3_query_expansion.md`.

1. First pass: top `fb_docs` documents from BM25.
2. RM1: `P(w|R) ∝ Σ_D P(w|D) · weight(D)`, weight = normalised first-pass score.
3. Filter candidates by document frequency, keep the top `fb_terms`.
4. RM3: `P'(w|Q) = (1−λ)·P_mle(w|Q) + λ·P(w|R)`.
5. Second pass with the expanded query as per-term weights.

Evaluation risks:

- **Pooling bias.** Judging only BM25's top results counts RM3's new documents as
  non-relevant. Label the union of both systems' top-K.
- **Tuning on the test set.** With a small query set, either split queries into
  tuning and test halves or report the full parameter sweep.

## 2. Resource-linking engine (branch `rm3`) — implemented

See `docs/resource_linking.md`.

1. **Fetch once and cache** each cited URL into `data/resources/` (README for GitHub
   and Hugging Face; title, meta description and visible text otherwise).
2. **Thin pages** are topped up with the text around the link in the citing proposal.
3. **Vectorize**: TF-IDF fitted on the proposal corpus; resources projected into it.
4. **Link**: top-k proposals by cosine similarity above a threshold, excluding the
   proposals that cite the resource.
5. **Export**: `resource_links.jsonl` and `proposal_related_resources.jsonl`.

First run on the snapshot: 19 of 25 resources fetched with usable text; 16 were linked
to at least one proposal (59 links, 14 with cosine ≥ 0.15).

## 3. Core MR

1. One integration branch off `dev`; teammates open PRs into it.
2. A pytest suite that runs without Docker (`backend/tests/test_ir.py` and
   `test_resources.py` are the pattern).
3. Small committed fixtures, since `data/` is gitignored.
4. README section on reproducing the pipeline from the snapshot.
5. New behaviour is additive and off by default (`expand=rm3`, resource links panel).
6. A single MR upstream once the integration branch is green.

## Interfaces to agree with the team

- Tokenizer and stemming (shared by indexing, queries, LDA and TF-IDF).
- Ranker surface: `search({term: weight}, k) -> [(doc_id, score)]` plus term statistics.
- Judgment files: tab-separated `query_id`, `doc_id`, `relevance`.
