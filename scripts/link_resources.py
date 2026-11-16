#!/usr/bin/env python3
"""Link cached external resources to topically similar proposals.

Reads the proposal corpus and the cache written by scripts/fetch_resources.py,
then writes two views of the same links:

    data/processed/resource_links.jsonl           one row per resource -> related proposals
    data/processed/proposal_related_resources.jsonl   one row per proposal -> related resources

Usage:
    python3 scripts/link_resources.py
    python3 scripts/link_resources.py --top-k 3 --min-score 0.15 --show
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ir import load_corpus, load_stopwords, tokenize  # noqa: E402
from app.ir.resources import boilerplate_descriptions, citations, link_context, link_resources, resource_tokens  # noqa: E402
from app.ir.tfidf import TfidfModel  # noqa: E402


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Link external resources to similar proposals.")
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data/processed/proposals_preprocessed.jsonl"))
    parser.add_argument("--resources", default=str(REPO_ROOT / "data/resources/resources.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data/processed/resource_links.jsonl"))
    parser.add_argument("--proposal-output",
                        default=str(REPO_ROOT / "data/processed/proposal_related_resources.jsonl"))
    parser.add_argument("--top-k", type=int, default=5, help="Proposals linked per resource")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum cosine similarity")
    parser.add_argument("--max-citing", type=int, default=5,
                        help="Ignore URLs cited by more proposals than this (template placeholders)")
    parser.add_argument("--allow-context-only", action="store_true",
                        help="Also link resources that could not be fetched, using only the text "
                             "around the link in the citing proposal (noisier)")
    parser.add_argument("--show", action="store_true", help="Print the links")
    args = parser.parse_args()

    docs, records = load_corpus(args.corpus)
    resources_path = Path(args.resources)
    if not resources_path.exists():
        sys.exit("%s not found; run scripts/fetch_resources.py first." % resources_path)
    pages = {}
    for line in resources_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            page = json.loads(line)
            pages[page["url"]] = page

    # Site-wide meta descriptions (e.g. YouTube's) say nothing about the page itself.
    boilerplate = boilerplate_descriptions(pages)
    for page in pages.values():
        if (page.get("description") or "").strip() in boilerplate:
            page["description"] = ""

    stopwords = load_stopwords()
    resources = []
    for url, cited_by in sorted(citations(records).items()):
        if len(cited_by) > args.max_citing:
            continue
        page = pages.get(url)
        contexts = [link_context(records[doc_id].get("description_raw", ""), url) for doc_id in cited_by]
        tokens, text_source = resource_tokens(page, contexts, lambda text: tokenize(text, stopwords),
                                              allow_context_only=args.allow_context_only)
        resources.append({
            "url": url,
            "title": (page or {}).get("title", ""),
            "text_source": text_source,
            "cited_by": cited_by,
            "tokens": tokens,
        })

    linked = link_resources(TfidfModel(docs), resources, top_k=args.top_k, min_score=args.min_score)

    resource_rows, by_proposal = [], {}
    for resource in linked:
        links = [
            {"submission_id": doc_id, "title": records[doc_id].get("title", ""), "score": round(score, 4)}
            for doc_id, score in resource["links"]
        ]
        resource_rows.append({
            "url": resource["url"],
            "title": resource["title"],
            "text_source": resource["text_source"],
            "token_count": len(resource["tokens"]),
            "cited_by": resource["cited_by"],
            "related_proposals": links,
        })
        for link in links:
            by_proposal.setdefault(link["submission_id"], []).append(
                {"url": resource["url"], "title": resource["title"], "score": link["score"]}
            )

    proposal_rows = [
        {"submission_id": doc_id, "related_resources": sorted(items, key=lambda r: (-r["score"], r["url"]))}
        for doc_id, items in sorted(by_proposal.items())
    ]
    write_jsonl(Path(args.output), resource_rows)
    write_jsonl(Path(args.proposal_output), proposal_rows)

    if args.show:
        for row in resource_rows:
            print("\n%s  [%s, %d tokens]\n  %s" % (row["url"], row["text_source"], row["token_count"], row["title"]))
            for link in row["related_proposals"]:
                print("    %.3f  %s" % (link["score"], link["title"][:90]))
            if not row["related_proposals"]:
                print("    (no proposal above the similarity threshold)")

    with_links = sum(1 for row in resource_rows if row["related_proposals"])
    print("\n%d resources, %d linked to at least one proposal; %d proposals received links."
          % (len(resource_rows), with_links, len(proposal_rows)))
    print("Wrote %s and %s" % (args.output, args.proposal_output))


if __name__ == "__main__":
    main()
