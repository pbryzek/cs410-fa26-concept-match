"""Unit tests for TF-IDF and resource linking. No network or services needed.

Run with:  pytest backend/tests/test_resources.py
"""

import math

import pytest

from app.ir.resources import (
    boilerplate_descriptions,
    citations,
    extract_page_text,
    is_login_redirect,
    link_context,
    link_resources,
    readme_url,
    resource_tokens,
)
from app.ir.tfidf import TfidfModel, cosine

DOCS = {
    "p1": "recipe search ingredients cooking".split(),
    "p2": "recipe recommendation ingredients food".split(),
    "p3": "legal document retrieval legislation".split(),
    "p4": "book reviews recommendation".split(),
}


def simple_tokenize(text):
    return text.lower().split()


# --- TF-IDF ------------------------------------------------------------------

def test_vectors_are_unit_length():
    model = TfidfModel(DOCS)
    for vector in model.doc_vectors.values():
        assert math.sqrt(sum(w * w for w in vector.values())) == pytest.approx(1.0)


def test_cosine_identity_and_disjoint():
    model = TfidfModel(DOCS)
    assert cosine(model.doc_vectors["p1"], model.doc_vectors["p1"]) == pytest.approx(1.0)
    assert cosine(model.doc_vectors["p1"], model.doc_vectors["p3"]) == 0.0


def test_rare_terms_weigh_more_than_common_terms():
    model = TfidfModel(DOCS)
    vector = model.vectorize(["recipe", "cooking"])  # "recipe" df=2, "cooking" df=1
    assert vector["cooking"] > vector["recipe"]


def test_unknown_terms_are_dropped():
    model = TfidfModel(DOCS)
    assert model.vectorize(["zzz"]) == {}
    assert set(model.vectorize(["zzz", "recipe"])) == {"recipe"}


def test_cosine_is_symmetric():
    model = TfidfModel(DOCS)
    a, b = model.doc_vectors["p1"], model.doc_vectors["p2"]
    assert cosine(a, b) == pytest.approx(cosine(b, a))
    assert 0 < cosine(a, b) < 1


# --- page text extraction ----------------------------------------------------

HTML = """
<html><head>
  <title>  Plain   Title </title>
  <meta name="description" content="A page about recipes.">
  <meta property="og:title" content="OG Title">
  <script>var hidden = "tracking";</script>
  <style>.hidden { color: red }</style>
</head><body>
  <nav>Home About</nav>
  <h1>Cooking &amp; Food</h1>
  <p>Visible paragraph.</p>
  <footer>Copyright</footer>
</body></html>
"""


def test_extract_page_text_reads_title_description_and_body():
    page = extract_page_text(HTML)
    assert page["title"] == "OG Title"
    assert page["description"] == "A page about recipes."
    assert page["text"] == "Cooking & Food Visible paragraph."


def test_extract_page_text_falls_back_to_title_tag_and_handles_empty():
    assert extract_page_text("<title> Plain   Title </title>")["title"] == "Plain Title"
    assert extract_page_text("") == {"title": "", "description": "", "text": ""}
    assert extract_page_text(None)["text"] == ""


def test_extract_page_text_truncates():
    assert len(extract_page_text("<p>" + "word " * 100 + "</p>", max_chars=50)["text"]) == 50


# --- URL helpers -------------------------------------------------------------

def test_readme_url():
    assert readme_url("https://github.com/owner/repo/tree/main/data") == \
        "https://api.github.com/repos/owner/repo/readme"
    assert readme_url("https://huggingface.co/datasets/allenai/scifact") == \
        "https://huggingface.co/datasets/allenai/scifact/raw/main/README.md"
    assert readme_url("https://huggingface.co/org/model") == \
        "https://huggingface.co/org/model/raw/main/README.md"
    assert readme_url("https://github.com/") is None
    assert readme_url("https://example.com/a/b") is None


def test_is_login_redirect():
    assert is_login_redirect("https://org.sharepoint.com/x", "https://login.microsoftonline.com/common/oauth2")
    assert not is_login_redirect("https://youtu.be/abc", "https://www.youtube.com/watch?v=abc")
    assert not is_login_redirect("https://example.com/login", "https://example.com/login")


def test_boilerplate_descriptions():
    pages = {
        "a": {"description": "Enjoy the videos."},
        "b": {"description": "Enjoy the videos."},
        "c": {"description": "Specific text."},
        "d": {"description": ""},
    }
    assert boilerplate_descriptions(pages) == {"Enjoy the videos."}


def test_link_context_takes_surrounding_text_and_strips_urls():
    text = "Dataset link: https://example.com/data (see also https://other.org/x) for recipes"
    context = link_context(text, "https://example.com/data")
    assert context == "Dataset link: (see also for recipes"
    assert link_context(text, "https://missing.example") == ""
    assert link_context(None, "https://example.com") == ""


def test_citations():
    records = {
        "p2": {"external_urls": ["https://a.example"]},
        "p1": {"external_urls": ["https://a.example", "https://b.example"]},
        "p3": {},
    }
    assert citations(records) == {"https://a.example": ["p1", "p2"], "https://b.example": ["p1"]}


# --- resource representation -------------------------------------------------

def test_resource_tokens_weights_title():
    page = {"ok": True, "title": "recipe", "description": "food", "text": "cooking"}
    tokens, source = resource_tokens(page, [], simple_tokenize, title_weight=2, min_page_tokens=1)
    assert tokens == ["recipe", "recipe", "food", "cooking"]
    assert source == "page"


def test_resource_tokens_tops_up_thin_pages_with_context():
    page = {"ok": True, "title": "api", "description": "", "text": ""}
    tokens, source = resource_tokens(page, ["federal bill search"], simple_tokenize, min_page_tokens=20)
    assert tokens[-3:] == ["federal", "bill", "search"]
    assert source == "page+context"


def test_resource_tokens_skips_unfetched_unless_allowed():
    failed = {"ok": False, "title": "Page not found", "text": "error"}
    assert resource_tokens(failed, ["recipe search"], simple_tokenize) == ([], "unfetched")
    assert resource_tokens(None, ["recipe search"], simple_tokenize) == ([], "unfetched")
    assert resource_tokens(None, ["recipe search"], simple_tokenize, allow_context_only=True) == \
        (["recipe", "search"], "context")


# --- linking -----------------------------------------------------------------

def test_link_resources_ranks_similar_proposals_and_excludes_citing_ones():
    model = TfidfModel(DOCS)
    resource = {"url": "https://r.example", "tokens": "recipe ingredients cooking".split(), "cited_by": ["p1"]}
    linked = link_resources(model, [resource], top_k=5, min_score=0.05)[0]
    ids = [doc_id for doc_id, _ in linked["links"]]
    assert ids == ["p2"]              # p1 cites it; p3 and p4 share no terms
    assert linked["url"] == "https://r.example"


def test_link_resources_applies_top_k_and_threshold():
    model = TfidfModel(DOCS)
    resource = {"url": "u", "tokens": "recipe recommendation ingredients".split(), "cited_by": []}
    scores = dict(link_resources(model, [resource], top_k=5, min_score=0.0)[0]["links"])
    assert scores["p2"] > scores["p1"] > 0
    assert len(link_resources(model, [resource], top_k=1, min_score=0.0)[0]["links"]) == 1
    assert link_resources(model, [resource], top_k=5, min_score=0.99)[0]["links"] == []


def test_link_resources_handles_resource_with_no_terms():
    model = TfidfModel(DOCS)
    assert link_resources(model, [{"url": "u", "tokens": [], "cited_by": []}])[0]["links"] == []
