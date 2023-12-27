"""Tests for solr dsl helpers."""
import pytest
from typing import List
from test_utils import w_scenarios
import pandas as pd
import numpy as np

from searcharray.solr import parse_min_should_match, edismax
from searcharray.postings import PostingsArray


def test_standard_percentage():
    assert parse_min_should_match(10, "50%") == 5


def test_over_100_percentage():
    assert parse_min_should_match(10, "150%") == 10


def test_negative_percentage():
    assert parse_min_should_match(10, "-50%") == 5


def test_standard_integer():
    assert parse_min_should_match(10, "3") == 3


def test_negative_integer():
    assert parse_min_should_match(10, "-3") == 7


def test_integer_exceeding_clause_count():
    assert parse_min_should_match(10, "15") == 10


def test_conditional_spec_less_than_clause_count():
    assert parse_min_should_match(10, "5<70%") == 7


def test_conditional_spec_greater_than_clause_count():
    assert parse_min_should_match(10, "15<70%") == 10


def test_complex_conditional_spec():
    assert parse_min_should_match(10, "3<50% 5<30%") == 3


def test_invalid_spec_percentage():
    with pytest.raises(ValueError):
        parse_min_should_match(10, "five%")


def test_invalid_spec_integer():
    with pytest.raises(ValueError):
        parse_min_should_match(10, "five")


def test_invalid_spec_conditional():
    with pytest.raises(ValueError):
        parse_min_should_match(10, "5<")


def test_empty_spec():
    with pytest.raises(ValueError):
        parse_min_should_match(10, "")


def test_complex_conditional_spec_with_percentage():
    assert parse_min_should_match(10, "2<2 5<3 7<40%") == 4


def everythings_a_b_tokenizer(text: str) -> List[str]:
    """Split on whitespace and return a list of tokens."""
    return ["b"] * len(text.split())


edismax_scenarios = {
    "base": {
        "frame": {
            'title': lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: PostingsArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"])
        },
        "expected": [lambda frame: sum([frame['title'].array.bm25("foo")[0],
                                        frame['title'].array.bm25("bar")[0]]),
                     0,
                     lambda frame: max(frame['title'].array.bm25("bar")[2],
                                       frame['body'].array.bm25("bar")[2]),
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"]},
    },
    "pf_title": {
        "frame": {
            'title': lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: PostingsArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"])
        },
        "expected": [lambda frame: sum([frame['title'].array.bm25(["foo", "bar"])[0],
                                        frame['title'].array.bm25("foo")[0],
                                        frame['title'].array.bm25("bar")[0]]),
                     0,
                     lambda frame: max(frame['title'].array.bm25("bar")[2],
                                       frame['body'].array.bm25("bar")[2]),
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"],
                   'pf': ["title"]}
    },
    "different_analyzers": {
        "frame": {
            'title': lambda: PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: PostingsArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"],
                                                tokenizer=everythings_a_b_tokenizer)
        },
        "expected": [lambda frame: max(frame['title'].array.bm25("bar")[0],
                                       frame['body'].array.bm25("b")[0]),

                     lambda frame: frame['body'].array.bm25("b")[1],

                     lambda frame: max(frame['title'].array.bm25("bar")[2],
                                       frame['body'].array.bm25("b")[2]),

                     lambda frame: frame['body'].array.bm25("b")[3]],
        "params": {'q': "bar", 'qf': ["title", "body"]},
    },
}


def build_df(frame):
    for k, v in frame.items():
        if hasattr(v, '__call__'):
            frame[k] = v()
    frame = pd.DataFrame(frame)
    return frame


def compute_expected(expected, frame):
    for idx, exp in enumerate(expected):
        if hasattr(exp, '__call__'):
            comp_expected = exp(frame)
            yield comp_expected
        else:
            yield exp


@w_scenarios(edismax_scenarios)
def test_edismax(frame, expected, params):
    frame = build_df(frame)
    expected = list(compute_expected(expected, frame))
    scores = edismax(frame, **params)
    assert np.allclose(scores, expected)
