from searcharray.postings import SearchArray
from test_utils import w_scenarios
import pytest
from searcharray.phrase.middle_out import MAX_POSN
import numpy as np


def random_strings(num_strings, min_length, max_length):
    strings = []
    for _ in range(num_strings):
        length = np.random.randint(min_length, max_length)
        string = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
        strings.append(string)
    return strings


scenarios = {
    "length_one": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "base": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "term_does_not_exist": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["term_does", "not_exist"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "and_but_not_phrase": {
        "docs": lambda: SearchArray.index(["foo bear bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "term_repeats": {
        "docs": lambda: SearchArray.index(["foo foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "multi_term_one_doc": {
        "docs": lambda: SearchArray.index(["foo bar bar bar foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "three_terms_match": {
        "docs": lambda: SearchArray.index(["foo bar baz baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "three_terms_no_match": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "three_terms_spread_out": {
        "docs": lambda: SearchArray.index(["foo bar EEK foo URG bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "same_term_matches": {
        "docs": lambda: SearchArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches": {
        "docs": lambda: SearchArray.index(["foo foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches_tail": {
        "docs": lambda: SearchArray.index(["foo bar bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "partial_same_term_matches_multiple": {
        "docs": lambda: SearchArray.index(["foo bar bar foo bar bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "bar"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "same_term_matches_3": {
        "docs": lambda: SearchArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "same_term_matches_4": {
        "docs": lambda: SearchArray.index(["foo foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo", "foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "same_term_phrase_repeats": {
        "docs": lambda: SearchArray.index(["foo foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "same_term_phrase_repeats_with_break": {
        "docs": lambda: SearchArray.index(["foo foo foo foo baz foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [3, 0, 0, 0] * 25,
    },
    "2x_same_term": {
        "docs": lambda: SearchArray.index(["foo foo bar bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "bar", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "duplicate_phrases": {
        "docs": lambda: SearchArray.index(["foo bar foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "duplicate_three_term_phrases": {
        "docs": lambda: SearchArray.index(["foo bar baz foo bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [2, 0, 0, 0] * 25,
    },
    "duplicate_three_term_phrases_last_disconnects": {
        "docs": lambda: SearchArray.index(["foo bar baz foo bar buzz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "different_num_posns": {
        "docs": lambda: SearchArray.index(["foo " + " ".join(["bar"] * 50),
                                          "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "different_num_posns_fewer": {
        "docs": lambda: SearchArray.index(["foo " + " ".join(["bar"] * 5),
                                          "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "different_num_posns_mixed": {
        "docs": lambda: SearchArray.index(["foo " + " ".join(["bar"] * 5),
                                           "foo " + " ".join(["bar"] * 50),
                                           "data2",
                                           "data3 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 1, 0, 0, 0] * 25,
    },
    "different_num_posns_mixed_and_not_phrase": {
        "docs": lambda: SearchArray.index(["data3 bar bar foo foo",
                                           "foo " + " ".join(["bar"] * 5),
                                           "foo " + " ".join(["bar"] * 50),
                                           "foo data2 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [0, 1, 1, 0, 0] * 25,
    },
    "long_doc": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                           "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0, 2] * 25
    },
    "long_phrase": {
        "docs": lambda: SearchArray.index(["foo la ma bar bar baz", "data2 ma ta", "data3 bar ma", "bunny funny wunny",
                                           "la ma ta wa ga ao a b c d e f g a be ae i la ma ta wa ga ao a foo bar foo bar"] * 25),
        "phrase": ["la", "ma", "ta", "wa", "ga", "ao", "a"],
        "expected": [0, 0, 0, 0, 2] * 25
    },
    "many_phrases": {
        "docs": lambda: SearchArray.index(["foo bar bar baz "
                                           + " ".join([" dummy foo bar baz"] * 100),
                                           "data2", "data3 bar", "bunny funny wunny foo bar"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [101, 0, 0, 1] * 25,
    },
    "10k_docs": {
        "docs": lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 10000),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 10000,
    },
}


@w_scenarios(scenarios)
def test_phrase_api(docs, phrase, expected):
    docs = docs()
    docs_before = docs.copy()
    term_freqs = docs.termfreqs(phrase)
    expected_matches = np.array(expected) > 0
    matches = docs.match(phrase)
    assert (term_freqs == expected).all()
    assert (matches == expected_matches).all()
    assert (docs == docs_before).all()


@w_scenarios(scenarios)
@pytest.mark.parametrize("algorithm", ["phrase_freq", "phrase_freq_scan",
                                       "phrase_freq_every_diff"])
def test_phrase(docs, phrase, expected, algorithm):
    # if np.all(expected[:5] == [0, 1, 1, 0, 0]) and algorithm in ["phrase_freq_scan", "phrase_freq_scan_inplace"]:
    #     pytest.skip("phrase_freq_scan known failure - different_num_posns_mixed_and_not_phrase")
    #     return

    docs = docs()
    docs_before = docs.copy()
    if len(phrase) > 1:
        phrase_matches = getattr(docs, algorithm)(phrase)
        assert (expected == phrase_matches).all()
        assert (docs == docs_before).all()
        phrase_matches2 = getattr(docs_before, algorithm)(phrase)
        assert (expected == phrase_matches2).all()


@pytest.mark.parametrize("phrase", ["foo bar baz", "foo bar",
                                    "foo foo foo", "foo foo bar",
                                    "foo bar bar",
                                    "foo bar bar baz buz foo bar",
                                    "foo bar bar baz buz foo foo",
                                    "foo foo", "foo foo bar", "foo bar bar"])
@pytest.mark.parametrize("posn_offset", range(100))
def test_phrase_different_posns(posn_offset, phrase):
    docs = SearchArray.index([" ".join(["dummy"] * posn_offset) + " " + phrase,
                             "not match"])
    phrase = phrase.split()
    expected = [1, 0]
    phrase_matches = docs.phrase_freq(phrase)
    assert (expected == phrase_matches).all()


@pytest.mark.parametrize("posn_offset", range(100))
def test_phrase_scattered_posns(posn_offset):
    scattered = "foo bar " + " ".join(["dummy"] * posn_offset) + " foo bar baz"
    docs = SearchArray.index([scattered,
                             "not match"])
    phrase = ["foo", "bar"]
    expected = [2, 0]
    phrase_matches = docs.phrase_freq(phrase)
    assert (expected == phrase_matches).all()


@pytest.mark.parametrize("posn_offset", range(100))
def test_phrase_scattered_posns3(posn_offset):
    scattered = "foo bar baz " + " ".join(["dummy"] * posn_offset) + " foo bar baz"
    docs = SearchArray.index([scattered,
                             "not match"])
    phrase = ["foo", "bar", "baz"]
    expected = [2, 0]
    phrase_matches = docs.phrase_freq(phrase)
    assert (expected == phrase_matches).all()


def test_phrase_too_many_posns():
    big_str = "foo bar baz " + " ".join(["dummy"] * MAX_POSN) + " foo bar baz"
    with pytest.raises(ValueError):
        SearchArray.index([big_str, "not match"])


def test_phrase_too_many_posns_with_truncate():
    big_str = "foo bar baz " + " ".join(["dummy"] * MAX_POSN) + " blah blah blah"
    arr = SearchArray.index([big_str, "not match"], truncate=True)
    assert len(arr) == 2
    phrase_matches = arr.phrase_freq(["foo", "bar", "baz"])
    expected = [1, 0]
    assert (expected == phrase_matches).all()


def test_positions():
    data = SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    positions = data.positions("bar")
    for idx, posn in enumerate(positions):
        if idx % 4 == 0:
            assert (posn == [1, 2]).all()
        elif idx % 4 == 2:
            assert (posn == [1]).all()
        else:
            assert (posn == []).all()


def test_positions_mask():
    data = SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    positions = data.positions("bar", np.asarray([True, False, False, False] * 25))
    assert len(positions) == 25
    for idx, posn in enumerate(positions):
        assert (posn == [1, 2]).all()


def test_positions_mask_single():
    data = SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"])
    positions = data.positions("bar", np.asarray([True, False, False, False]))
    assert len(positions) == 1
    for idx, posn in enumerate(positions):
        assert (posn == [1, 2]).all()
