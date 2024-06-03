from pandas.tests.extension import base
import pandas as pd
import pytest


from searcharray import SearchArray, Terms, TermsDtype


@pytest.fixture
def dtype():
    return TermsDtype()


@pytest.fixture
def data():
    """Return a fixture of your data here that returns an instance of your ExtensionArray."""
    return SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)


@pytest.fixture
def data_missing():
    """Return a fixture of your data with missing values here."""
    return SearchArray.index(["", "foo bar baz"])


@pytest.fixture
def na_cmp():
    return lambda x, y: x == Terms({}) or y == Terms({})


@pytest.fixture
def na_value():
    return Terms({})


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
    arr = SearchArray.index(["abba mmma dabbb", "abba abba aska", "caa cata"])
    return arr


@pytest.fixture
def data_missing_for_sorting():
    arr = SearchArray.index(["abba mmma dabbb", "", "caa cata"])
    return arr


@pytest.fixture
def data_for_grouping():
    """Get data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    arr = SearchArray.index(["abba mmma dabbb", "abba mmma dabbb",
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

    # Unique not supported on inverted index rows, for performance
    # reasons
    def test_value_counts_with_normalize(self, data):
        pass

    def test_unique(self, data):
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
