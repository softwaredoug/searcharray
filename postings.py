"""Tokenized, searchable text as a pandas dtype."""
import pandas as pd
import numbers
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype
from pandas.api.types import is_list_like
from pandas.api.extensions import take

import numpy as np
from term_dict import TermDict

# Doc,Term -> freq
from scipy.sparse import lil_matrix, csr_matrix


class PostingsDtype(ExtensionDtype):
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
        return PostingsArray

    def __repr__(self):
        return 'PostingsDtype()'

    @property
    def na_value(self):
        return None

    def valid_value(self, value):
        return isinstance(value, str) or pd.isna(value)


register_extension_dtype(PostingsDtype)


def ws_tokenizer(string):
    if pd.isna(string):
        return []
    if not isinstance(string, str):
        raise ValueError("Expected a string")
    return string.split()


class PostingsArray(ExtensionArray):
    dtype = PostingsDtype()

    def __init__(self, strings, tokenizer=ws_tokenizer):
        # Check dtype, raise TypeError
        if not is_list_like(strings):
            raise TypeError("Expected list-like object, got {}".format(type(strings)))
        if not all(isinstance(x, str) or pd.isna(x) for x in strings):
            raise TypeError("Expected a list of strings")

        freqs_table = lil_matrix((len(strings), 0))
        self.term_dict = TermDict()
        self.avg_doc_length = 0
        for doc_id, string in enumerate(strings):
            if pd.isna(string):
                continue
            tokenized = tokenizer(string)
            self.avg_doc_length += len(tokenized)
            for token in tokenized:
                term_id = self.term_dict.add_term(token)
                if term_id >= freqs_table.shape[1]:
                    freqs_table.resize((freqs_table.shape[0], term_id + 1))
                freqs_table[doc_id, term_id] += 1

        self.term_freqs = csr_matrix(freqs_table)
        self.avg_doc_length /= len(strings)

        # How to eliminate data?
        self.data = np.asarray(strings, dtype=object)
        self.tokenizer = tokenizer

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if dtype is not None:
            if not isinstance(dtype, PostingsDtype):
                return scalars
        if type(scalars) == np.ndarray and scalars.dtype == PostingsDtype():
            return cls(scalars)
        # String types
        elif type(scalars) == np.ndarray and scalars.dtype.kind in 'US':
            return cls(scalars)
        # Other objects
        elif type(scalars) == np.ndarray and scalars.dtype != object:
            return scalars
        return cls(scalars)

    def memory_usage(self, deep=False):
        return self.nbytes

    @property
    def nbytes(self):
        all_bytes = [len(x) for x in self.data]
        return sum(all_bytes)

    def __getitem__(self, key):
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(key, int):
            return self.data[key]
        else:
            return PostingsArray(self.data[key], tokenizer=self.tokenizer)

    def __setitem__(self, key, value):
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, PostingsArray):
            value = value.data
        if isinstance(value, list):
            value = np.asarray(value, dtype=object)

        if not isinstance(value, np.ndarray) and not self.dtype.valid_value(value):
            raise ValueError(f"Cannot set non-object array to PostingsArray -- you passed type:{type(value)} -- {value}")

        # Cant set a single value to an array
        if isinstance(key, numbers.Integral) and isinstance(value, np.ndarray):
            raise ValueError("Cannot set a single value to an array")

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
        if isinstance(other, PostingsArray):
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
        return PostingsArray(result, tokenizer=self.tokenizer)

    def copy(self):
        return PostingsArray(self.data.copy(), tokenizer=self.tokenizer)

    @classmethod
    def _concat_same_type(cls, to_concat):
        concatenated_data = np.concatenate([ea.data for ea in to_concat])
        return PostingsArray(concatenated_data, tokenizer=to_concat[0].tokenizer)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def _values_for_factorize(self):
        arr = self.data.copy()
        return arr, None

    # ***********************************************************
    # Naive implementations of search functions to clean up later
    # ***********************************************************
    def term_freq(self, tokenized_term):
        if not isinstance(tokenized_term, str):
            raise TypeError("Expected a string")

        term_id = self.term_dict.get_term_id(tokenized_term)
        matches = self.term_freqs[:, term_id].todense().flatten()
        matches = np.asarray(matches).flatten()
        return matches

    def doc_freq(self, tokenized_term):
        if not isinstance(tokenized_term, str):
            raise TypeError("Expected a string")
        # Count number of rows where the term appears
        term_freq = self.term_freq(tokenized_term)
        return np.sum(term_freq > 0)

    def doc_lengths(self):
        return np.sum(self.term_freqs, axis=1)

    def match(self, tokenized_term):
        """Return a boolean numpy array indicating which elements contain the given term."""
        term_freq = self.term_freq(tokenized_term)
        return term_freq > 0

    def bm25_idf(self, tokenized_term):
        df = self.doc_freq(tokenized_term)
        num_docs = len(self)
        return np.log(1 + (num_docs - df + 0.5) / (df + 0.5))

    def bm25_tf(self, tokenized_term, k1=1.2, b=0.75):
        tf = self.term_freq(tokenized_term)
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (self.doc_lengths() / self.avg_doc_length)))

    def bm25(self, tokenized_term, k1=1.2, b=0.75):
        """Score each doc using BM25."""
        import pdb; pdb.set_trace()
        return self.bm25_idf(tokenized_term) * self.bm25_tf(tokenized_term)
