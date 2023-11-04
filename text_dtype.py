"""Tokenized, searchable text as a pandas dtype."""
import pandas as pd
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.api.types import is_list_like
from pandas.api.extensions import take
import numpy as np


class TokenizedTextDtype(ExtensionDtype):
    name = 'tokenized_text'
    type = str
    kind = 'O'  # Object kind

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return TokenizedTextArray

    def __repr__(self):
        return 'TokenizedTextDtype()'

    @property
    def na_value(self):
        return None


def ws_tokenizer(string):
    if pd.isna(string):
        return []
    if not isinstance(string, str):
        raise ValueError("Expected a string")
    return string.split()


class TokenizedTextArray(ExtensionArray):
    dtype = TokenizedTextDtype()

    def __init__(self, strings, tokenizer=ws_tokenizer):
        # Check dtype, raise TypeError
        if not is_list_like(strings):
            raise TypeError("Expected list-like object, got {}".format(type(strings)))
        if not all(isinstance(x, str) or pd.isna(x) for x in strings):
            raise TypeError("Expected a list of strings")

        self.data = np.asarray(strings, dtype=object)
        self.tokenizer = tokenizer

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if dtype is not None:
            if not isinstance(dtype, TokenizedTextDtype):
                return scalars
        if type(scalars) == np.ndarray and scalars.dtype != object:
            return scalars
        return cls(scalars)

    def memory_usage(self, deep=False):
        # This is required for Series/DataFrame.info() to work
        return self.nbytes

    def nbytes(self):
        return sum(len(x) for x in self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        else:
            return TokenizedTextArray(self.data[idx])

    def __setitem__(self, key, value):
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, TokenizedTextArray):
            value = value.data
        self.data[key] = value

    def value_counts(
        self,
        dropna: bool = True,
    ):
        from collections import Counter

        if dropna:
            counts = Counter(self.data)
            counts.pop(None, None)
        else:
            counts = Counter(self.data)
        return pd.Series(counts)

    def __len__(self):
        return len(self.data)

    def __ne__(self, other):
        if isinstance(other, pd.DataFrame) or isinstance(other, pd.Series) or isinstance(other, pd.Index):
            return NotImplemented

        return ~(self == other)

    def __eq__(self, other):
        """Return a boolean numpy array indicating elementwise equality."""
        # When other is a dataframe or series, not implemented
        if isinstance(other, pd.DataFrame) or isinstance(other, pd.Series) or isinstance(other, pd.Index):
            return NotImplemented

        # When other is an ExtensionArray
        if isinstance(other, TokenizedTextArray):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            return np.array([self.tokenizer(a) == self.tokenizer(b) for a, b in zip(self.data, other.data)])

        # When other is a scalar value
        elif isinstance(other, str):
            return np.array([self.tokenizer(a) == self.tokenizer(other) for a in self.data])

        # When other is a sequence but not an ExtensionArray
        elif is_list_like(other):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            return np.array([self.tokenizer(a) == self.tokenizer(b) for a, b in zip(self.data, other)])

        # Return False where 'other' is neither the same length nor a scalar
        else:
            return np.full(len(self), False)

    def isna(self):
        return np.array([pd.isna(x) for x in self.data], dtype=bool)

    def take(self, indices, allow_fill=False, fill_value=None):

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None
        return TokenizedTextArray(result)

    def copy(self):
        return TokenizedTextArray(self.data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        concatenated_data = np.concatenate([ea.data for ea in to_concat])
        return TokenizedTextArray(concatenated_data)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def _values_for_factorize(self):
        arr = self.data.copy()
        return arr, None

    # Example method for token-based searching
    def contains_token(self, token):
        return np.array([token in self.tokenizer(text) for text in self.data])
