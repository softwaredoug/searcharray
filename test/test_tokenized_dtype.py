import pytest
from pandas.tests.extension import base
import pandas as pd

from text_dtype import TokenizedTextDtype, TokenizedTextArray


@pytest.fixture
def dtype():
    return TokenizedTextDtype()


@pytest.fixture
def data():
    """Return a fixture of your data here that returns an instance of your ExtensionArray."""
    return TokenizedTextArray(["foo bar baz", "data2", "data3", "bunny funny wunny"] * 25)


@pytest.fixture
def data_missing():
    """Return a fixture of your data with missing values here."""
    return TokenizedTextArray([None, "foo bar baz"])


@pytest.fixture
def na_cmp():
    return lambda x, y: x is None and y is None


@pytest.fixture
def na_value():
    return None


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
    return TokenizedTextArray(["mmma dabbb", "zed bar bar", "aaa bb aa"])


@pytest.fixture
def data_missing_for_sorting():
    return TokenizedTextArray(["mmma dabbb", None, "aaa bb aa"])


@pytest.fixture
def data_for_grouping():
    """Get data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return TokenizedTextArray(["foo bar baz", "foo bar baz", None, None, "abba cadabra", "abba cadabra",
                               "foo bar baz", "zunny funny wunny"])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
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
