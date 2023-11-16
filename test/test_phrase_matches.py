from searcharray.postings import PostingsArray
from test_utils import w_scenarios
from time import perf_counter
import pytest
import numpy as np


def random_strings(num_strings, min_length, max_length):
    strings = []
    for _ in range(num_strings):
        length = np.random.randint(min_length, max_length)
        string = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
        strings.append(string)
    return strings


perf_scenarios = {
    "1m_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 1000000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 1000000,
    },
    "10m_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 10000000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 10000000,
    },
    "many_docs_long_doc": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                             "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False, True] * 100000,
    },
    "many_docs_large_term_dict": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                             " ".join(random_strings(1000, 4, 10)),
                                             "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False, False, True] * 100000,
    },
}


scenarios = {
    "base": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "multi_term_one_doc": {
        "docs": lambda: PostingsArray.index(["foo bar bar bar foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "three_terms_match": {
        "docs": lambda: PostingsArray.index(["foo bar baz baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "three_terms_no_match": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "three_terms_spread_out": {
        "docs": lambda: PostingsArray.index(["foo bar EEK foo URG bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "same_term_matches": {
        "docs": lambda: PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches": {
        "docs": lambda: PostingsArray.index(["foo foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches_tail": {
        "docs": lambda: PostingsArray.index(["foo bar bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches_multiple": {
        "docs": lambda: PostingsArray.index(["foo bar bar foo bar bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "bar"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "same_term_matches_3": {
        "docs": lambda: PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "same_term_matches_4": {
        "docs": lambda: PostingsArray.index(["foo foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "same_term_phrase_repeats": {
        "docs": lambda: PostingsArray.index(["foo foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "same_term_phrase_repeats_with_break": {
        "docs": lambda: PostingsArray.index(["foo foo foo foo baz foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [3, 0, 0, 0] * 25,
    },
    "duplicate_phrases": {
        "docs": lambda: PostingsArray.index(["foo bar foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "duplicate_three_term_phrases": {
        "docs": lambda: PostingsArray.index(["foo bar baz foo bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "duplicate_three_term_phrases_last_disconnects": {
        "docs": lambda: PostingsArray.index(["foo bar baz foo bar buzz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "10k_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 10000),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 10000,
    },
}


@w_scenarios(scenarios)
def test_phrase(docs, phrase, expected):
    docs = docs()
    docs_before = docs.copy()
    term_freqs = docs.phrase_freq(phrase)
    expected_matches = np.array(expected) > 0
    matches = docs.phrase_match(phrase)
    assert (term_freqs == expected).all()
    assert (matches == expected_matches).all()
    assert (docs == docs_before).all(), "The phrase_match method should not modify the original array"


@pytest.mark.skip
@w_scenarios(perf_scenarios)
def test_phrase_performance(docs, phrase, expected):
    start = perf_counter()
    docs = docs()
    matches = docs.phrase_match(phrase)
    print(f"phrase_match took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches == expected).all()


def test_positions():
    data = PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    positions = data.positions("bar")
    for idx, posn in enumerate(positions):
        if idx % 4 == 0:
            assert (posn == [1, 2]).all()
        elif idx % 4 == 2:
            assert (posn == [1]).all()
        else:
            assert (posn == []).all()
