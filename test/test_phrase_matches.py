from searcharray.postings import PostingsArray
from test_utils import w_scenarios
from time import perf_counter


scenarios = {
    "base": {
        "docs": PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "multi_term_one_doc": {
        "docs": PostingsArray.index(["foo bar bar bar foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "three_terms_match": {
        "docs": PostingsArray.index(["foo bar baz baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [True, False, False, False] * 25,
    },
    "three_terms_no_match": {
        "docs": PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [False, False, False, False] * 25,
    },
    "many_docs": {
        "docs": PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 100000,
    },
    "three_terms_spread_out": {
        "docs": PostingsArray.index(["foo bar EEK foo URG bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [False, False, False, False] * 25,
    },
    "same_term_matches": {
        "docs": PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [True, False, False, False] * 25,
    },
    "same_term_matches_3": {
        "docs": PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo"],
        "expected": [True, False, False, False] * 25,
    },
    "duplicate_phrases": {
        "docs": PostingsArray.index(["foo bar foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
}


@w_scenarios(scenarios)
def test_phrase(docs, phrase, expected):
    start = perf_counter()
    docs_before = docs.copy()
    matches = docs.phrase_match(phrase)
    print(f"phrase_match took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches == expected).all()
    if len(docs) < 1000:
        assert (docs == docs_before).all(), "The phrase_match method should not modify the original array"
