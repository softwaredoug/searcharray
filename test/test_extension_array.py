import pytest
import numpy as np
from pandas.tests.extension import base
import pandas as pd
import gzip
from time import perf_counter
import json

from postings import PostingsDtype, PostingsArray, PostingsRow
from test_utils import w_scenarios


@pytest.fixture
def tmdb_raw_data():
    path = 'fixtures/tmdb.json.gz'
    with gzip.open(path) as f:
        return json.load(f)


@pytest.fixture
def tmdb_data(tmdb_raw_data):
    ids = tmdb_raw_data.keys()
    titles = []
    overviews = []
    for id in ids:
        try:
            titles.append(tmdb_raw_data[id]['title'])
        except KeyError:
            titles.append('')

        try:
            overviews.append(tmdb_raw_data[id]['overview'])
        except KeyError:
            overviews.append('')

    assert len(ids) == len(titles) == len(overviews)

    df = pd.DataFrame({'title': titles, 'overview': overviews}, index=ids)
    indexed = PostingsArray.index(df['title'])
    df['title_tokens'] = indexed

    indexed = PostingsArray.index(df['overview'])
    df['overview_tokens'] = indexed
    return df


@pytest.fixture
def dtype():
    return PostingsDtype()


@pytest.fixture
def data():
    """Return a fixture of your data here that returns an instance of your ExtensionArray."""
    return PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)


@pytest.fixture
def data_missing():
    """Return a fixture of your data with missing values here."""
    return PostingsArray.index(["", "foo bar baz"])


@pytest.fixture
def na_cmp():
    return lambda x, y: x == PostingsRow({}) or y == PostingsRow({})


@pytest.fixture
def na_value():
    return PostingsRow({})


@pytest.fixture(params=[True, False])
def as_series(request):
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    return request.param


@pytest.fixture
def data_repeated(data):

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def invalid_scalar(data):
    return 123


@pytest.fixture(params=[True, False])
def use_numpy(request):
    return request.param


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    arr = PostingsArray.index(["abba mmma dabbb", "abba abba aska", "caa cata"])
    return arr


@pytest.fixture
def data_missing_for_sorting():
    arr = PostingsArray.index(["abba mmma dabbb", "", "caa cata"])
    return arr


@pytest.fixture
def data_for_grouping():
    """Get data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    arr = PostingsArray.index(["abba mmma dabbb", "abba mmma dabbb",
                               "", "",
                               "caa cata", "caa cata",
                               "abba mmma dabbb", "abba abba aska"])
    return arr


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    return request.param


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    return request.param


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Return whether to box the data in a Series."""
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """Return whether to support ExtensionDtype _from_sequence method testing."""
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    return request.param


# Then create a class that inherits from the base tests you want to use
class TestDType(base.BaseDtypeTests):
    # You'll need to at least provide the following attributes
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetItem(base.BaseGetitemTests):
    pass


class TestSetItem(base.BaseSetitemTests):
    pass


class TestCasting(base.BaseCastingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    pass


def test_match(data):
    matches = data.match("foo")
    assert (matches == [True, False, False, False] * 25).all()


def test_term_freqs(data):
    matches = data.term_freq("bar")
    assert (matches == [2, 0, 1, 0] * 25).all()


def test_doc_freq(data):
    doc_freq = data.doc_freq("bar")
    assert doc_freq == (2 * 25)


def test_bm25(data):
    bm25_idf = data.bm25_idf("bar")
    assert bm25_idf > 0.0
    bm25 = data.bm25("bar")
    assert bm25.shape == (100,)
    assert np.isclose(bm25, [1.60518294, 0.0 , 1.38629436, 0.0] * 25).all()


def test_positions(data):
    positions = data.positions("bar")
    for idx, posn in enumerate(positions):
        if idx % 4 == 0:
            assert posn == [1, 2]
        elif idx % 4 == 2:
            assert posn == [1]
        else:
            assert posn == []


def test_phrase_match(data):
    matches = data.phrase_match(["foo", "bar"])
    assert (matches == [True, False, False, False] * 25).all()


def test_phrase_match_three_terms(data):
    matches = data.phrase_match(["bunny", "funny", "wunny"])
    assert (matches == [False, False, False, True] * 25).all()


def test_tokenize_tmdb(tmdb_raw_data):
    ids = tmdb_raw_data.keys()
    titles = []
    overviews = []
    for id in ids:
        try:
            titles.append(tmdb_raw_data[id]['title'])
        except KeyError:
            titles.append('')

        try:
            overviews.append(tmdb_raw_data[id]['overview'])
        except KeyError:
            overviews.append('')

    assert len(ids) == len(titles) == len(overviews)

    df = pd.DataFrame({'title': titles, 'overview': overviews}, index=ids)
    # Create tokenized versions of each
    start = perf_counter()
    print("Indexing title...")
    indexed = PostingsArray.index(df['title'])
    stop = perf_counter()
    df['title_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    start = perf_counter()
    print("Indexing overview...")
    indexed = PostingsArray.index(df['overview'])
    stop = perf_counter()
    df['overview_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    assert len(df) == len(ids)


tmdb_phrase_matches = [
    (["Star", "Wars"], ['11', '330459', '76180']),
]


@pytest.mark.parametrize("phrase,expected_matches", tmdb_phrase_matches)
def test_phrase_match_tmdb(phrase, expected_matches, tmdb_data):
    mask = tmdb_data['title_tokens'].array.phrase_match(['Star', 'Wars'])
    matches = tmdb_data[mask].index.sort_values()
    assert (matches == expected_matches).all()
