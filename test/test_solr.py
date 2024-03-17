"""Tests for solr dsl helpers."""
import pytest
from typing import List
from test_utils import w_scenarios
import pandas as pd
import numpy as np

from searcharray.similarity import ScoringContext
from searcharray.solr import parse_min_should_match, edismax
from searcharray.postings import SearchArray


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


def just_lowercasing_tokenizer(text: str) -> List[str]:
    """Lowercase and return a list of tokens."""
    return [text.lower()]


edismax_scenarios = {
    "base": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"])
        },
        "expected": [lambda frame: sum([frame['title'].array.score("foo")[0],
                                        frame['title'].array.score("bar")[0]]),
                     0,
                     lambda frame: max(frame['title'].array.score("bar")[2],
                                       frame['body'].array.score("bar")[2]),
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"]},
    },
    "field_centric": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["foo bar", "data2", "data3 bar", "bunny funny wunny"],
                                              tokenizer=just_lowercasing_tokenizer)
        },
        "expected": [lambda frame: max(sum([frame['title'].array.score("foo")[0],
                                           frame['title'].array.score("bar")[0]]),
                                       frame['body'].array.score("foo bar")[0]),
                     0,
                     lambda frame: frame['title'].array.score("bar")[2],
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"]},
    },
    "field_centric_mm": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["foo bar", "data2", "data3 bar", "bunny funny wunny"],
                                              tokenizer=just_lowercasing_tokenizer)
        },
        "expected": [lambda frame: max(sum([frame['title'].array.score("foo")[0],
                                           frame['title'].array.score("bar")[0]]),
                                       frame['body'].array.score("foo bar")[0]),
                     0,
                     0,
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"], 'mm': "2"},
    },
    "field_centric_mm_opp": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["foo bar", "data2", "data3 bar", "bunny funny wunny"],
                                              tokenizer=just_lowercasing_tokenizer)
        },
        "expected": [lambda frame: max(sum([frame['title'].array.score("foo")[0],
                                           frame['title'].array.score("bar")[0]]),
                                       frame['body'].array.score("foo bar")[0]),
                     0,
                     0,
                     0],
        "params": {'q': "foo bar", 'qf': ["body", "title"], 'mm': "2"},
    },
    "boost_title": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"])
        },
        "expected": [lambda frame: sum([frame['title'].array.score("foo")[0] * 10,
                                        frame['title'].array.score("bar")[0] * 10]),
                     0,
                     lambda frame: max(frame['title'].array.score("bar")[2] * 10,
                                       frame['body'].array.score("bar")[2]),
                     0],
        "params": {'q': "foo bar", 'qf': ["title^10", "body"]},
    },
    "pf_title": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"])
        },
        "expected": [lambda frame: sum([frame['title'].array.score(["foo", "bar"])[0],
                                        frame['title'].array.score("foo")[0],
                                        frame['title'].array.score("bar")[0]]),
                     0,
                     lambda frame: max(frame['title'].array.score("bar")[2],
                                       frame['body'].array.score("bar")[2]),
                     0],
        "params": {'q': "foo bar", 'qf': ["title", "body"],
                   'pf': ["title"]}
    },
    "different_analyzers": {
        "frame": {
            'title': lambda: SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"]),
            'body': lambda: SearchArray.index(["buzz", "data2", "data3 bar", "bunny funny wunny"],
                                              tokenizer=everythings_a_b_tokenizer)
        },
        "expected": [lambda frame: max(frame['title'].array.score("bar")[0],
                                       frame['body'].array.score("b")[0]),

                     lambda frame: frame['body'].array.score("b")[1],

                     lambda frame: max(frame['title'].array.score("bar")[2],
                                       frame['body'].array.score("b")[2]),

                     lambda frame: frame['body'].array.score("b")[3]],
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
    scores, explain = edismax(frame, **params)
    assert np.allclose(scores, expected)


def always_one_similarity(term_freqs: np.ndarray, doc_freqs: np.ndarray, context: ScoringContext) -> np.ndarray:
    term_freqs = term_freqs
    return term_freqs > 0


@w_scenarios(edismax_scenarios)
def test_edismax_custom_similarity(frame, expected, params):
    frame = build_df(frame)
    expected = list(compute_expected(expected, frame))
    params['similarity'] = always_one_similarity
    scores, explain = edismax(frame, **params)
    assert np.all(scores.astype(np.int64) == scores)
