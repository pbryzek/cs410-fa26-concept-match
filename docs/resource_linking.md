# Resource Linking

Links the external URLs cited in proposals (repositories, datasets, videos) to
*other* proposals on similar topics, using cosine similarity over TF-IDF vectors.

## Where the code lives

| Path | Purpose |
| --- | --- |
| `backend/app/ir/tfidf.py` | TF-IDF model fitted on proposals, cosine similarity |
| `backend/app/ir/resources.py` | Page text extraction, resource representation, linking |
| `scripts/fetch_resources.py` | Fetch each cited URL once and cache its text |
| `scripts/link_resources.py` | Compute links and write the output files |
| `backend/tests/test_resources.py` | Unit tests (no network needed) |

## Pipeline

```bash
python3 scripts/export_proposals.py
python3 scripts/preprocess_proposals.py
python3 scripts/fetch_resources.py          # network; writes data/resources/resources.jsonl
python3 scripts/link_resources.py --show    # offline; writes the two files below
```

1. **Collect** the `external_urls` of every proposal. URLs cited by more than
   `--max-citing` proposals (default 5) are skipped: these are form-template
   placeholders such as `https://github.com/...`.
2. **Fetch once and cache.** For GitHub repositories and Hugging Face datasets the
   README is used; other pages contribute their title, meta description and
   visible text. Fetches that are bounced to a sign-in page count as failed.
   The cache is a static snapshot; re-running only fetches new URLs.
3. **Represent** each resource as terms: title (counted twice), description and
   text, tokenized like the proposals. Meta descriptions shared by several pages
   (site-wide boilerplate) are ignored. Pages with fewer than 20 terms are topped
   up with the text around the link in the citing proposal.
4. **Vectorize** with TF-IDF fitted on the proposal corpus. Resource terms the
   proposals never use are dropped.
5. **Link** each resource to its `--top-k` most similar proposals (default 5) with
   cosine similarity of at least `--min-score` (default 0.1). Proposals that
   already cite the resource are excluded.

Resources that could not be fetched get no links by default. `--allow-context-only`
links them using only the surrounding proposal text; this is noisier because that
text is often template wording shared by many proposals.

## Outputs

`data/processed/resource_links.jsonl`, one row per resource:

```json
{"url": "...", "title": "...", "text_source": "page", "token_count": 333,
 "cited_by": ["<submission_id>"],
 "related_proposals": [{"submission_id": "...", "title": "...", "score": 0.395}]}
```

`text_source` is `page`, `page+context`, `context` or `unfetched`.

`data/processed/proposal_related_resources.jsonl`, one row per proposal that
received at least one link, for merging into the JSON export by `submission_id`:

```json
{"submission_id": "...", "related_resources": [{"url": "...", "title": "...", "score": 0.395}]}
```

## Tests

```bash
.venv/bin/pytest backend/tests/test_resources.py
```
