from searcharray.postings import SearchArray
from test_utils import w_scenarios


scenarios = {
    "base": {
        "docs": lambda: SearchArray.index(["foo bar bar baz" + " ".join(["boz"] * 25) + " foo bar",
                                           "data2", "data3 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "min_posn": 0,
        "max_posn": 17,
        "expected": [1, 0, 0, 0] * 25,
    },
    "no_max": {
        "docs": lambda: SearchArray.index(["foo bar bar baz" + " ".join(["boz"] * 25) + " foo bar",
                                           "data2", "data3 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "min_posn": 0,
        "max_posn": None,
        "expected": [2, 0, 0, 0] * 25,
    },
    "no_max_min_18": {
        "docs": lambda: SearchArray.index(["foo bar bar baz" + " ".join(["boz"] * 25) + " foo bar",
                                           "data2", "data3 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "bar"],
        "min_posn": 18,
        "max_posn": None,
        "expected": [1, 0, 0, 0] * 25,
    },
    "same_term": {
        "docs": lambda: SearchArray.index(["foo foo baz baz" + " ".join(["boz"] * 25) + " foo foo",
                                           "data2", "data3 bar",
                                           "bunny funny wunny"] * 25),
        "phrase": ["foo", "foo"],
        "min_posn": 0,
        "max_posn": 17,
        "expected": [1, 0, 0, 0] * 25,
    },
}


@w_scenarios(scenarios)
def test_min_max(docs, phrase, min_posn, max_posn, expected):
    docs = docs()
    docs_before = docs.copy()
    term_freqs = docs.termfreqs(phrase, min_posn=min_posn,
                                max_posn=max_posn)
    assert (term_freqs == expected).all()
    assert (docs == docs_before).all()
