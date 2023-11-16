from searcharray.postings import PostingsArray
from scipy.sparse import csr_matrix
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
    "base": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "multi_term_one_doc": {
        "docs": lambda: PostingsArray.index(["foo bar bar bar foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "three_terms_match": {
        "docs": lambda: PostingsArray.index(["foo bar baz baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [True, False, False, False] * 25,
    },
    "three_terms_no_match": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [False, False, False, False] * 25,
    },
    "100k_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 100000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 100000,
    },
    "1m_docs": {
        "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 1000000),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 1000000,
    },
    # "10m_docs": {
    #    "docs": lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 10000000),
    #    "phrase": ["foo", "bar"],
    #    "expected": [True, False, False, False] * 10000000,
    # },
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
    "three_terms_spread_out": {
        "docs": lambda: PostingsArray.index(["foo bar EEK foo URG bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [False, False, False, False] * 25,
    },
    "same_term_matches": {
        "docs": lambda: PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "expected": [True, False, False, False] * 25,
    },
    "same_term_matches_3": {
        "docs": lambda: PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo", "foo"],
        "expected": [True, False, False, False] * 25,
    },
    "duplicate_phrases": {
        "docs": lambda: PostingsArray.index(["foo bar foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "expected": [True, False, False, False] * 25,
    },
    "duplicate_three_term_phrases": {
        "docs": lambda: PostingsArray.index(["foo bar baz foo bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [True, False, False, False] * 25,
    },
    "duplicate_three_term_phrases_last_disconnects": {
        "docs": lambda: PostingsArray.index(["foo bar baz foo bar buzz", "data2", "data3 bar", "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar", "baz"],
        "expected": [True, False, False, False] * 25,
    },
}


@w_scenarios(scenarios)
def test_phrase(docs, phrase, expected):
    start = perf_counter()
    docs = docs()
    docs_before = docs.copy()
    matches = docs.phrase_match(phrase)
    print(f"phrase_match took {perf_counter() - start} seconds | {len(docs)} docs")
    assert (matches == expected).all()
    if len(docs) < 1000:
        assert (docs == docs_before).all(), "The phrase_match method should not modify the original array"


def vstack_with_pad(arrays, width=5, pad_value=0):
    # First, create an array ofi len(arrays)xwidth
    vstacked = np.zeros((len(arrays), width), dtype=arrays[0].dtype)
    # Then, for each array, replace the first width values with the array's values
    for idx, array in enumerate(arrays):
        vstacked[idx, :len(array)] = array
    return vstacked
    # Then, vstack the arrays


@pytest.mark.skip
def test_slice_stack_mat_posns():
    mat = csr_matrix([[1, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1]])
    nonzeros = mat.nonzero()
    mat[nonzeros] = (nonzeros[1] + 1)

    cols = np.max(np.diff(mat.indptr))

    # indptr -> indexed into the indices
    # indices -> nonzero column indices


    # Mask out up to C column of nonzero values from mat
    C = 3
    mask = np.zeros(mat.shape, dtype=bool)
    mask[:, :C] = True
    import pdb; pdb.set_trace()
    new_mat = mat.todense()
    new_mat = mat[mask][mat != 0]

    import pdb; pdb.set_trace()


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
