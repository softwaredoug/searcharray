"""Tokenized, searchable text as a pandas dtype"""
import pandas as pd
from pandas.api.extensions import ExtensionDtype, ExtensionArray
import numpy as np


class TokenizedTextDtype(ExtensionDtype):
    name = 'tokenized_text'
    type = str
    kind = 'O'  # Object kind

    @classmethod
    def construct_from_string(cls, string):
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    def __repr__(self):
        return 'TokenizedTextDtype()'


class TokenizedTextArray(ExtensionArray):
    dtype = TokenizedTextDtype()

    def __init__(self, strings, tokenize_on=' '):
        self.data = np.asarray(strings, dtype=object)
        self.tokenize_on = tokenize_on
        # Here, we are not actually tokenizing the string yet,
        # but you could tokenize in the initializer if needed.

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Tokenize on the fly when an item is accessed.
            return self.data[idx].split(self.tokenize_on)
        else:
            # If idx is not a single integer, return a new TokenizedTextArray
            return TokenizedTextArray(self.data[idx])

    def __len__(self):
        return len(self.data)

    def isna(self):
        return np.array([pd.isna(x) for x in self.data], dtype=bool)

    def take(self, indices, allow_fill=False, fill_value=None):
        taken = np.take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return TokenizedTextArray(taken)

    def copy(self):
        return TokenizedTextArray(self.data.copy())

    def _concat_same_type(self, to_concat):
        concatenated_data = np.concatenate([ea.data for ea in to_concat])
        return TokenizedTextArray(concatenated_data)

    # Example method for token-based searching
    def contains_token(self, token):
        return np.array([token in text.split(self.tokenize_on) for text in self.data])
