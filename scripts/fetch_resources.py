#!/usr/bin/env python3
"""Fetch the external URLs cited in proposals once and cache their text.

The cache is a static snapshot, like the proposal export: later steps read it
and never touch the network. Re-running only fetches URLs that are not cached
yet (use --refresh to fetch everything again).

Usage:
    python3 scripts/fetch_resources.py
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ir.corpus import load_corpus  # noqa: E402
from app.ir.resources import citations, extract_page_text, is_login_redirect, readme_url  # noqa: E402

USER_AGENT = "Mozilla/5.0 (compatible; resource-linker/1.0)"
MAX_BYTES = 3_000_000
MAX_TEXT_CHARS = 20000


def http_get(url: str, timeout: float, accept: str = "text/html,*/*"):
    """Return (status, body text, final URL). Network failures come back as status 0."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_BYTES).decode(charset, errors="replace")
            return response.status, body, response.geturl()
    except urllib.error.HTTPError as error:
        return error.code, "", url
    except Exception as error:  # DNS failure, timeout, TLS error, bad URL
        return 0, str(error), url


def fetch_resource(url: str, timeout: float) -> dict:
    record = {"url": url, "ok": False, "status": 0, "title": "", "description": "", "text": "", "error": ""}

    status, body, final_url = http_get(url, timeout)
    record["status"] = status
    if status == 200 and is_login_redirect(url, final_url):
        record["error"] = "redirected to a sign-in page"
    elif status == 200:
        record.update(extract_page_text(body, MAX_TEXT_CHARS))
        record["ok"] = True
    elif status == 0:
        record["error"] = body

    # Repository and dataset pages are mostly chrome; the README is the real content.
    readme = readme_url(url)
    if readme:
        readme_status, readme_body, _ = http_get(readme, timeout, accept="application/vnd.github.raw, text/plain")
        if readme_status == 200 and readme_body.strip():
            record["text"] = readme_body[:MAX_TEXT_CHARS]
            record["ok"] = True
            if not record["title"]:
                record["title"] = url.rstrip("/").split("/")[-1]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache the text of external URLs cited in proposals.")
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data/processed/proposals_preprocessed.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data/resources/resources.jsonl"))
    parser.add_argument("--max-citing", type=int, default=5,
                        help="Skip URLs cited by more proposals than this (template placeholders)")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between requests")
    parser.add_argument("--refresh", action="store_true", help="Fetch again even if already cached")
    args = parser.parse_args()

    _, records = load_corpus(args.corpus)
    cited = citations(records)
    urls = sorted(url for url, ids in cited.items() if len(ids) <= args.max_citing)
    skipped = sorted(url for url, ids in cited.items() if len(ids) > args.max_citing)

    output = Path(args.output)
    cache = {}
    if output.exists() and not args.refresh:
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["url"]] = row

    for url in urls:
        if url in cache:
            continue
        cache[url] = fetch_resource(url, args.timeout)
        cache[url]["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print("%-4s %6d chars  %s" % (cache[url]["status"] or "ERR", len(cache[url]["text"]), url))
        time.sleep(args.delay)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for url in sorted(cache):
            handle.write(json.dumps(cache[url], ensure_ascii=False) + "\n")

    usable = sum(1 for url in urls if cache[url]["ok"])
    print("\n%d URLs cached in %s (%d with usable text, %d failed)." % (len(urls), output, usable, len(urls) - usable))
    for url in skipped:
        print("Skipped placeholder cited by %d proposals: %s" % (len(cited[url]), url))


if __name__ == "__main__":
    main()
