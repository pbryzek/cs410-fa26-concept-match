"""Unit tests for the offline retrieval package. No server, Mongo or OpenSearch needed.

Run with:  pytest backend/tests/test_ir.py
"""

import math

import pytest

from app.ir.bm25 import BM25Index
from app.ir.metrics import average_precision, ndcg_at_k, precision_at_k
from app.ir.rm3 import RM3Config, expand_query, relevance_model, rm3_search
from app.ir.tokenizer import load_stopwords, tokenize

# "rocchio" and "feedback" never share a query term with d3, but d1 and d2 tie
# them together, which is the vocabulary-mismatch case RM3 is meant to fix.
DOCS = {
    "d1": "relevance feedback rocchio query expansion".split(),
    "d2": "relevance feedback rocchio pseudo".split(),
    "d3": "rocchio algorithm vector space".split(),
    "d4": "topic model lda gibbs sampling".split(),
    "d5": "topic model plsa em algorithm".split(),
    "d6": "sentiment analysis movie reviews".split(),
}

# Small corpus, so turn the document-frequency filters off unless a test needs them.
OPEN = dict(min_df=1, max_df_ratio=1.0)


@pytest.fixture
def index():
    return BM25Index(DOCS)


# --- tokenizer ---------------------------------------------------------------

def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Rocchio Feedback-Loop, v2!", {"the"}) == ["rocchio", "feedback-loop", "v2"]


def test_tokenize_handles_empty_input():
    assert tokenize("", {"the"}) == []
    assert tokenize(None) == []


def test_default_stopwords_load():
    assert "the" in load_stopwords()


# --- BM25 --------------------------------------------------------------------

def test_bm25_ranks_matching_documents_only(index):
    ranked = [doc_id for doc_id, _ in index.search(["topic", "model"], k=None)]
    assert set(ranked) == {"d4", "d5"}


def test_bm25_score_matches_formula(index):
    # "lda" occurs once, in d4 only; every document has 5 tokens except d2, d3, d6 (4).
    idf = math.log(1 + (6 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - 0.75 + 0.75 * 5 / index.avg_doc_length
    expected = idf * 1 * (1.2 + 1) / (1 + 1.2 * norm)
    assert index.search(["lda"]) == [("d4", pytest.approx(expected))]


def test_bm25_rare_term_outweighs_common_term(index):
    # d5 matches the rare "plsa"; d4 matches only the more common "topic".
    assert index.search(["topic", "plsa"])[0][0] == "d5"


def test_bm25_query_weights_change_ranking(index):
    assert index.search({"lda": 1.0, "plsa": 0.1})[0][0] == "d4"
    assert index.search({"lda": 0.1, "plsa": 1.0})[0][0] == "d5"


def test_bm25_term_list_equals_count_weights(index):
    assert index.search(["topic", "topic", "lda"]) == index.search({"topic": 2, "lda": 1})


def test_bm25_unknown_terms_and_empty_index():
    assert BM25Index(DOCS).search(["zzz"]) == []
    assert BM25Index({}).search(["topic"]) == []


def test_bm25_respects_k(index):
    assert len(index.search(["rocchio"], k=2)) == 2
    assert len(index.search(["rocchio"], k=None)) == 3


# --- RM3 ---------------------------------------------------------------------

def test_relevance_model_is_a_distribution(index):
    model = relevance_model(index, index.search(["feedback"], 2))
    assert sum(model.values()) == pytest.approx(1.0)
    assert set(model) == set(DOCS["d1"]) | set(DOCS["d2"])


def test_expansion_adds_cooccurring_terms(index):
    result = expand_query(index, ["feedback"], RM3Config(fb_docs=2, fb_terms=5, **OPEN))
    new_terms = [term for term, _ in result.expansion_terms]
    assert "rocchio" in new_terms and "relevance" in new_terms
    assert "feedback" not in new_terms
    assert sorted(result.feedback_docs) == ["d1", "d2"]


def test_expanded_query_weights_sum_to_one(index):
    result = expand_query(index, ["feedback"], RM3Config(fb_docs=2, fb_terms=3, **OPEN))
    assert sum(result.expanded_query.values()) == pytest.approx(1.0)


def test_rm3_retrieves_document_bm25_misses(index):
    baseline = {doc_id for doc_id, _ in index.search(["feedback"], k=None)}
    expanded = {doc_id for doc_id, _ in rm3_search(index, ["feedback"], None,
                                                   RM3Config(fb_docs=2, **OPEN)).results}
    assert "d3" not in baseline
    assert "d3" in expanded


def test_zero_feedback_weight_equals_baseline(index):
    result = rm3_search(index, ["topic", "model"], 10, RM3Config(feedback_weight=0.0, **OPEN))
    assert result.expansion_terms == []
    assert [d for d, _ in result.results] == [d for d, _ in index.search(["topic", "model"], 10)]


def test_feedback_weight_one_still_searches(index):
    result = rm3_search(index, ["feedback"], 10, RM3Config(fb_docs=2, feedback_weight=1.0, **OPEN))
    assert result.results
    assert sum(result.expanded_query.values()) == pytest.approx(1.0)


def test_fb_terms_limits_feedback_model(index):
    result = expand_query(index, ["feedback"], RM3Config(fb_docs=2, fb_terms=2, **OPEN))
    assert len(result.expanded_query) <= 3  # 2 feedback terms plus the original query term


def test_df_filters_and_exclusions_drop_candidates_but_keep_query_terms(index):
    # "rocchio" is in 3 of 6 documents; a 0.4 cap removes it as an expansion term.
    capped = expand_query(index, ["feedback"], RM3Config(fb_docs=2, min_df=1, max_df_ratio=0.4))
    assert "rocchio" not in capped.expanded_query
    assert "feedback" in capped.expanded_query

    rare = expand_query(index, ["feedback"], RM3Config(fb_docs=2, min_df=2, max_df_ratio=1.0))
    assert "pseudo" not in rare.expanded_query  # df = 1

    excluded = expand_query(index, ["feedback"],
                            RM3Config(fb_docs=2, exclude_terms=frozenset({"rocchio"}), **OPEN))
    assert "rocchio" not in excluded.expanded_query

    kept = expand_query(index, ["rocchio"], RM3Config(fb_docs=2, min_df=1, max_df_ratio=0.1))
    assert "rocchio" in kept.expanded_query


def test_rm3_handles_empty_and_unmatched_queries(index):
    assert rm3_search(index, [], 10).results == []
    unmatched = rm3_search(index, ["zzz"], 10)
    assert unmatched.results == [] and unmatched.expansion_terms == []


# --- metrics -----------------------------------------------------------------

JUDGMENTS = {"a": 2, "b": 0, "c": 1, "d": 1}


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], JUDGMENTS, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], JUDGMENTS, 5) == pytest.approx(1 / 5)
    assert precision_at_k(["x", "y"], JUDGMENTS, 2) == 0.0


def test_average_precision():
    # Relevant at ranks 1 and 3; "d" is relevant but never retrieved.
    assert average_precision(["a", "b", "c"], JUDGMENTS) == pytest.approx((1 / 1 + 2 / 3) / 3)
    assert average_precision(["a"], {}) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "c", "d"], JUDGMENTS, 3) == pytest.approx(1.0)
    dcg = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(["b", "a", "c"], JUDGMENTS, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a"], {"a": 0}, 3) == 0.0
