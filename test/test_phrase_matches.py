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


scenarios = {
    "length_one": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "base": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "term_does_not_exist": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["term_does", "not_exist"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "and_but_not_phrase": {
        "docs": lambda: PostingsArray.index(["foo bear bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [0, 0, 0, 0] * 25,
    },
    "term_repeats": {
        "docs": lambda: PostingsArray.index(["foo foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
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
    "different_num_posns": {
        "docs": lambda: PostingsArray.index(["foo " + " ".join(["bar"] * 50),
                                             "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "different_num_posns_fewer": {
        "docs": lambda: PostingsArray.index(["foo " + " ".join(["bar"] * 5),
                                             "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 25,
    },
    "different_num_posns_mixed": {
        "docs": lambda: PostingsArray.index(["foo " + " ".join(["bar"] * 5),
                                             "foo " + " ".join(["bar"] * 50),
                                             "data2",
                                             "data3 bar",
                                             "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 1, 0, 0, 0] * 25,
    },
    "different_num_posns_mixed_and_not_phrase": {
        "docs": lambda: PostingsArray.index(["data3 bar bar foo foo",
                                             "foo " + " ".join(["bar"] * 5),
                                             "foo " + " ".join(["bar"] * 50),
                                             "foo data2 bar",
                                             "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [0, 1, 1, 0, 0] * 25,
    },
    "long_doc": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                             "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0, 2] * 25
    },
    "10k_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 10000),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0] * 10000,
    },
}


@w_scenarios(scenarios)
def test_phrase_api(docs, phrase, expected):
    docs = docs()
    docs_before = docs.copy()
    term_freqs = docs.term_freq(phrase)
    expected_matches = np.array(expected) > 0
    matches = docs.match(phrase)
    assert (term_freqs == expected).all()
    assert (matches == expected_matches).all()
    assert (docs == docs_before).all()


@w_scenarios(scenarios)
@pytest.mark.parametrize("algorithm", ["phrase_freq", "phrase_freq_scan_old",
                                       "phrase_freq_scan", "phrase_freq_scan_inplace", "phrase_freq_wide_spans"])
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


perf_scenarios = {
    "4m_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 1000000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 1000000,
    },
    "many_docs_long_doc": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                             "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0, 2] * 100000,
    },
    "many_docs_large_term_dict": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny",
                                             " ".join(random_strings(1000, 4, 10)),
                                             "la ma ta wa ga ao a b c d e f g a be ae i foo bar foo bar"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [1, 0, 0, 0, 0, 2] * 100000,
    },
    "many_docs_and_positions": {
        "docs": lambda: PostingsArray.index(["foo bar",
                                             " ".join(["foo bar bar baz foo foo bar foo"] * 100),
                                             " ".join(["what is the foo bar doing in the bar foo?"] * 100)] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [1, 200, 100] * 100000
    }

}


# phrase_match_every_diff  took 17.07792454198352 seconds | 200000 docs
# phrase_match_scan old    took 16.765271917014616 seconds | 200000 docs
# phrase_match_scan        took 81.19630783301545 seconds | 200000 docs
# phrase_match_scan        took 70.4959268750099 seconds | 200000 docs
#
# phrase_match_every_diff  took 2.214169082988519 seconds | 4000000 docs
# phrase_match_scan old    took 69.71960766700795 seconds | 4000000 docs
# phrase_match_scan        took 4.758700999984285 seconds | 4000000 docs
# phrase_match_scan        took 4.029075291007757 seconds | 4000000 docs

@pytest.mark.skip("perf")
@w_scenarios(perf_scenarios)
def test_phrase_performance(docs, phrase, expected):
    start = perf_counter()
    docs = docs()
    print(f"Indexing took {perf_counter() - start} seconds | {len(docs)} docs")

    print(f"Starting phrase: {phrase} -- expected: {expected[:10]}")

    start = perf_counter()
    matches = docs.phrase_freq(phrase)
    print(f"phrase_freq API took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches == expected).all()

    start = perf_counter()
    matches_every_diff = docs.phrase_freq_every_diff(phrase)
    print(f"phrase_match_every_diff  took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_every_diff == expected).all()

    start = perf_counter()
    matches_scan_old = docs.phrase_freq_scan_old(phrase)
    print(f"phrase_match_scan old    took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_scan_old == expected).all()

    start = perf_counter()
    matches_scan = docs.phrase_freq_scan(phrase)
    print(f"phrase_match_scan        took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_scan == expected).all()

    start = perf_counter()
    matches_scan_inplace = docs.phrase_freq_scan_inplace(phrase)
    print(f"phrase_match_scan inplace took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_scan_inplace == expected).all()

    start = perf_counter()
    matches_scan_inplace = docs.phrase_freq_wide_spans(phrase)
    print(f"phrase_match_scan widespa took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_scan_inplace == expected).all()

    start = perf_counter()
    matches_scan_inplace = docs.phrase_freq_scan_inplace_binsearch(phrase)
    print(f"phrase_match_scan inplbin took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches_scan_inplace == expected).all()


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


def test_positions_mask():
    data = PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    positions = data.positions("bar", np.asarray([True, False, False, False] * 25))
    assert len(positions) == 25
    for idx, posn in enumerate(positions):
        assert (posn == [1, 2]).all()


def test_positions_mask_single():
    data = PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"])
    positions = data.positions("bar", np.asarray([True, False, False, False]))
    assert len(positions) == 1
    for idx, posn in enumerate(positions):
        assert (posn == [1, 2]).all()
