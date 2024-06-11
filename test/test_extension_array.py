from pandas.tests.extension import base
import pandas as pd
import pytest

from searcharray import SearchArray, LazyTerms, TermsDtype


@pytest.fixture
def dtype():
    return TermsDtype()


@pytest.fixture
def data():
    """Return a fixture of your data here that returns an instance of your ExtensionArray."""
    arr = SearchArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    for idx, item in enumerate(arr):
        assert idx == item.doc_id
    return arr


@pytest.fixture
def data_missing():
    """Return a fixture of your data with missing values here."""
    return SearchArray.index(["", "foo bar baz"])


@pytest.fixture
def na_cmp():
    return lambda x, y: x == LazyTerms() or y == LazyTerms()


@pytest.fixture
def na_value():
    return LazyTerms()


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
    pytest.skip("Grouping not supported by SearchArray")
    """
    pass


@pytest.fixture
def data_missing_for_sorting():
    pytest.skip("Grouping not supported by SearchArray")


@pytest.fixture
def data_for_grouping():
    """Get data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pytest.skip("Grouping not supported by SearchArray")


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


def test_na_values_eq():
    assert LazyTerms() == LazyTerms()


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
        pytest.skip("Unique not supported on inverted index rows, for performance reasons")

    def test_unique(self, data):
        pytest.skip("Unique not supported on inverted index rows, for performance reasons")

    def test_argsort(self):
        pytest.skip("sorting not supported for inverted index rows")

    def test_argsort_missing(self):
        pytest.skip("sorting not supported for inverted index rows")

    def test_nargsort(self, data_for_sorting):
        pytest.skip("sorting not supported for inverted index rows")

    def test_sort_values_missing(self, data_missing_for_sorting):
        pytest.skip("sorting not supported for inverted index rows")

    def test_argsort_missing_array(self, data_missing_for_sorting):
        pytest.skip("sorting not supported for inverted index rows")

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        pytest.skip("sorting not supported for inverted index rows")

    def test_argmin_argmax(self, data):
        pytest.skip("argmin and argmax not supported for inverted index rows")

    def test_argmin_argmax_all_na(self, data_missing):
        pytest.skip("argmin and argmax not supported for inverted index rows")

    def test_argreduce_series(self, data):
        pytest.skip("argmin and argmax not supported for inverted index rows")

    def test_factorize_empty(self, data):
        pytest.skip("factorize not supported for inverted index rows")

    def test_searchsorted(self, data):
        pytest.skip("searchsorted not supported for inverted index rows")

    def test_sort_values_frame(self, data_for_sorting, sort_by_key):
        pytest.skip("sorting not supported for inverted index rows")




class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetItem(base.BaseGetitemTests):
    pass


class TestCasting(base.BaseCastingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass
