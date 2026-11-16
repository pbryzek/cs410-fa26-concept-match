"""Resource linking: connect external URLs cited in proposals to other, topically
similar proposals using cosine similarity over TF-IDF vectors.

Everything here is pure (no network); ``scripts/fetch_resources.py`` does the fetching.
"""

from html.parser import HTMLParser
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse
import re

from app.ir.tfidf import TfidfModel, cosine

URL_RE = re.compile(r"https?://\S+|www\.\S+")

SKIPPED_TAGS = {"script", "style", "noscript", "svg", "template", "nav", "header", "footer", "form"}
VOID_TAGS = {"meta", "link", "br", "hr", "img", "input", "source", "area", "base", "col", "embed", "wbr"}


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts: List[str] = []
        self.meta: Dict[str, str] = {}
        self.body_parts: List[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            attributes = dict(attrs)
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            if key in ("description", "og:description", "og:title") and attributes.get("content"):
                self.meta.setdefault(key, attributes["content"])
        elif tag == "title":
            self._in_title = True
        elif tag in SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip_depth and data.strip():
            self.body_parts.append(data.strip())


def extract_page_text(html: str, max_chars: int = 20000) -> Dict[str, str]:
    """Pull the title, meta description and visible text out of an HTML page."""
    parser = _PageParser()
    try:
        parser.feed(html or "")
    except Exception:  # malformed markup: keep whatever was parsed so far
        pass
    title = parser.meta.get("og:title") or " ".join(" ".join(parser.title_parts).split())
    description = parser.meta.get("description") or parser.meta.get("og:description") or ""
    return {
        "title": title.strip(),
        "description": description.strip(),
        "text": " ".join(parser.body_parts)[:max_chars],
    }


def readme_url(url: str) -> Optional[str]:
    """Raw README location for GitHub repositories and Hugging Face datasets or models."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host in ("github.com", "www.github.com") and len(parts) >= 2:
        return "https://api.github.com/repos/%s/%s/readme" % (parts[0], parts[1])
    if host == "huggingface.co":
        if len(parts) >= 3 and parts[0] in ("datasets", "spaces"):
            return "https://huggingface.co/%s/raw/main/README.md" % "/".join(parts[:3])
        if len(parts) == 2:
            return "https://huggingface.co/%s/raw/main/README.md" % "/".join(parts)
    return None


def is_login_redirect(requested_url: str, final_url: str) -> bool:
    """True when a fetch was bounced to a sign-in page instead of the content."""
    if urlparse(requested_url).netloc.lower() == urlparse(final_url).netloc.lower():
        return False
    final = urlparse(final_url)
    return bool(re.search(r"login|signin|sign-in|accounts\.|/auth", final.netloc.lower() + final.path.lower()))


def boilerplate_descriptions(pages: Mapping[str, dict]) -> set:
    """Meta descriptions shared by several pages (site-wide text, not about the page)."""
    seen: Dict[str, int] = {}
    for page in pages.values():
        description = (page.get("description") or "").strip()
        if description:
            seen[description] = seen.get(description, 0) + 1
    return {description for description, count in seen.items() if count > 1}


def link_context(text: str, url: str, window: int = 200) -> str:
    """Text surrounding each mention of ``url`` in a proposal, with URLs removed."""
    text = text or ""
    snippets = []
    start = text.find(url)
    while start != -1:
        before = text[max(0, start - window):start]
        after = text[start + len(url):start + len(url) + window]
        snippets.append(before + " " + after)
        start = text.find(url, start + len(url))
    return " ".join(URL_RE.sub(" ", " ".join(snippets)).split())


def citations(records: Mapping[str, dict]) -> Dict[str, List[str]]:
    """Map each external URL to the sorted ids of the proposals citing it."""
    cited: Dict[str, List[str]] = {}
    for doc_id, record in records.items():
        for url in record.get("external_urls") or []:
            cited.setdefault(url, []).append(doc_id)
    return {url: sorted(ids) for url, ids in cited.items()}


def resource_tokens(page: Optional[dict], contexts: List[str], tokenize: Callable[[str], List[str]],
                    title_weight: int = 2, min_page_tokens: int = 20,
                    allow_context_only: bool = False) -> Tuple[List[str], str]:
    """Build a resource's term list and say where it came from.

    Uses the fetched page (title counted ``title_weight`` times). A thin page is
    topped up with the text around the link in the citing proposals.

    A resource that could not be fetched at all gets no terms unless
    ``allow_context_only`` is set: link context alone is often just form-template
    wording ("Tech Review Video URL: ..."), which matches every proposal that
    used the same template.
    """
    tokens: List[str] = []
    if page and page.get("ok"):
        tokens += tokenize(page.get("title", "")) * title_weight
        tokens += tokenize(page.get("description", ""))
        tokens += tokenize(page.get("text", ""))
    source = "page" if tokens else ""
    if not tokens and not allow_context_only:
        return [], "unfetched"
    if len(tokens) < min_page_tokens:
        context_tokens = tokenize(" ".join(contexts))
        if context_tokens:
            tokens += context_tokens
            source = "page+context" if source else "context"
    return tokens, source or "none"


def link_resources(model: TfidfModel, resources: List[dict], top_k: int = 5,
                   min_score: float = 0.1) -> List[dict]:
    """Rank proposals for each resource by cosine similarity.

    Each resource is ``{"url", "tokens", "cited_by"}``. Proposals that cite the
    resource are excluded: they are already connected to it, and (when link
    context is used) they would match trivially.
    """
    linked = []
    for resource in resources:
        vector = model.vectorize(resource["tokens"])
        excluded = set(resource.get("cited_by") or [])
        scored = [
            (doc_id, cosine(vector, doc_vector))
            for doc_id, doc_vector in model.doc_vectors.items()
            if doc_id not in excluded
        ]
        scored = [(doc_id, score) for doc_id, score in scored if score >= min_score]
        scored.sort(key=lambda item: (-item[1], item[0]))
        linked.append(dict(resource, links=scored[:top_k]))
    return linked
