"""Test postings array search functionality."""
import numpy as np
import pytest
from searcharray.postings import SearchArray
from test_utils import w_scenarios


@pytest.fixture
def data():
    """Return a fixture of your data here that returns an instance of your ExtensionArray."""
    return SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)


def test_match(data):
    matches = data.match("foo")
    assert (matches == [True, False, False, False] * 25).all()


def test_match_missing_term(data):
    matches = data.match("not_present")
    assert (matches == [False, False, False, False] * 25).all()


def test_term_freqs(data):
    matches = data.termfreqs("bar")
    assert (matches == [2, 0, 1, 0] * 25).all()


def test_doc_freq(data):
    doc_freq = data.docfreq("bar")
    assert doc_freq == (2 * 25)
    doc_freq = data.docfreq("foo")
    assert doc_freq == 25


def test_doc_lengths(data):
    doc_lengths = data.doclengths()
    assert doc_lengths.shape == (100,)
    assert (doc_lengths == [4, 1, 2, 3] * 25).all()
    assert data.avg_doc_length == 2.5


def test_default_score_matches_lucene(data):
    bm25 = data.score("bar")
    assert bm25.shape == (100,)
    assert np.isclose(bm25, [0.37066694, 0., 0.34314217, 0.] * 25).all()


and_scenarios = {
    "base": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "no_match": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "data2"],
        "expected": [False, False, False, False] * 25,
    },
    "and_with_phrase": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": [["foo", "bar"], "baz"],
        "expected": [True, False, False, False] * 25,
    }
}


@w_scenarios(and_scenarios)
def test_and_query(data, docs, keywords, expected):
    docs = docs()
    matches = data.and_query(keywords)
    assert (expected == matches).all()


or_scenarios = {
    "base": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "bar"],
        "expected": [True, False, True, False] * 25,
        "min_should_match": 1,
    },
    "mm_2": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
        "min_should_match": 2,
    },
    "one_term_match": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "data2"],
        "expected": [True, True, False, False] * 25,
        "min_should_match": 1,
    },
    "one_term_match_mm2": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": ["foo", "data2"],
        "expected": [False, False, False, False] * 25,
        "min_should_match": 2,
    },
    "or_with_phrase": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": [["foo", "bar"], "baz"],
        "expected": [True, False, False, False] * 25,
        "min_should_match": 1,
    },
    "or_with_phrase_on_copy": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25, avoid_copies=False),
        "keywords": [["foo", "bar"], "baz"],
        "expected": [True, False, False, False] * 25,
        "min_should_match": 1,
    },
    "or_with_phrase_mm2": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "keywords": [["foo", "bar"], ["bar", "baz"]],
        "expected": [True, False, False, False] * 25,
        "min_should_match": 2,
    }
}


@w_scenarios(or_scenarios)
def test_or_query(docs, keywords, expected, min_should_match):
    docs = docs()
    matches = docs.or_query(keywords, min_should_match=min_should_match)
    assert (expected == matches).all()


@w_scenarios(or_scenarios)
def test_or_query_sliced(docs, keywords, expected, min_should_match):
    docs = docs()
    num_docs = len(docs)
    sliced = docs[:num_docs // 2]
    expected_sliced = expected[:num_docs // 2]
    matches = sliced.or_query(keywords, min_should_match=min_should_match)
    assert (expected_sliced == matches).all()


@w_scenarios(or_scenarios)
def test_or_query_copy(docs, keywords, expected, min_should_match):
    docs = docs()
    num_docs = len(docs)
    sliced = docs[:num_docs // 2].copy()
    expected_sliced = expected[:num_docs // 2]
    matches = sliced.or_query(keywords, min_should_match=min_should_match)
    assert (expected_sliced == matches).all()
