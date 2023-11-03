import pytest
from pandas.tests.extension import base

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
    return TokenizedTextArray(["foo bar baz", None, "data3", None] * 25)


# Then create a class that inherits from the base tests you want to use
class TestMyExtension(base.BaseDtypeTests, base.BaseInterfaceTests, base.BaseArithmeticOpsTests):
    # You'll need to at least provide the following attributes
    pass
