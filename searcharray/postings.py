"""Tokenized, searchable text as a pandas dtype."""
import pandas as pd
import numbers
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype
from pandas.api.types import is_list_like
from pandas.api.extensions import take
import json
from collections import Counter
import warnings
import logging
from typing import List, Union, Optional, Iterable


import numpy as np
from searcharray.phrase.middle_out import PosnBitArray
from searcharray.similarity import Similarity, default_bm25
from searcharray.indexing import build_index_from_tokenizer, build_index_from_terms_list
from searcharray.term_dict import TermMissingError
from searcharray.roaringish.roaringish_ops import as_dense

logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


def bytes_to_human_readable(num_bytes):
    """
    Convert a number of bytes into a human-readable format (KB, MB, GB, etc.)

    :param num_bytes: Number of bytes to be converted
    :type num_bytes: int or float
    :return: Human-readable string
    :rtype: str
    """
    # Define the suffixes for byte conversion
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

    if num_bytes == 0:
        return "0B"

    i = 0
    while num_bytes >= 1024 and i < len(suffixes) - 1:
        num_bytes /= 1024.
        i += 1

    return f"{num_bytes:.2f} {suffixes[i]}"


class Terms:
    """An indexed search doc - a single bag of tokenized words and positions."""

    def __init__(self,
                 postings,
                 doc_len: int = 0,
                 posns: Optional[dict] = None,
                 encoded=False):
        self.postings = postings
        self.posns = None
        self.encoded = encoded
        self.doc_len = doc_len
        self.posns = posns

    def _validate_posns(self):
        # (For testing/assertions) - Confirm every term in positions also in postings
        if self.posns is None:
            return
        for term in self.posns:
            if term not in self.postings:
                raise ValueError(f"Term {term} in positions but not in postings. ")

    def termfreq(self, token):
        return self.postings[token]

    def terms(self):
        return self.postings.items()

    def positions(self, term=None):
        if self.posns is None:
            return {}
        if term is None:
            posns = self.posns.items()
        else:
            posns = self.posns[term]
        return posns

    def raw_positions(self, term_dict, term=None):
        if self.posns is None:
            return {}
        if term is None:
            posns = [(term_dict.get_term_id(term), posns) for term, posns in self.posns.items()]
        else:
            posns = [(term_dict.get_term_id(term), self.posns[term])]
        return posns

    def tf_to_dense(self, term_dict):
        """Convert to a dense vector of term frequencies."""
        dense = np.zeros(len(term_dict))
        for term, freq in self.terms():
            dense[term_dict.get_term_id(term)] = freq
        return dense

    def __len__(self):
        return len(self.postings)

    def __repr__(self):
        posting_keys = set(self.postings.keys())
        rval = f"Terms({posting_keys})"
        return rval

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        # Flip to the other implementation if we're comparing to a SearchArray
        # to get a boolean array back
        if isinstance(other, SearchArray):
            return other == self
        same_postings = isinstance(other, Terms) and self.postings == other.postings
        if same_postings and self.doc_len == other.doc_len:
            return True

    def __lt__(self, other):
        # return isinstance(other, Terms) and hash(self) < hash(other)
        keys_both = set(self.postings.keys()).union(set(other.postings.keys()))
        # Sort lexically
        keys_both = sorted(keys_both)

        # Iterate as if these are two vectors of the same large dimensional vector sparse
        for key in keys_both:
            lhs_val = 0
            rhs_val = 0
            try:
                lhs_val = self.postings[key]
            except KeyError:
                pass

            try:
                rhs_val = other.postings[key]
            except KeyError:
                pass

            if lhs_val < rhs_val:
                return True
            elif lhs_val > rhs_val:
                return False
            else:
                continue
        return False

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other) and self != other

    def __hash__(self):
        return hash(json.dumps(self.postings, sort_keys=True))


class TermsDtype(ExtensionDtype):
    """Pandas dtype for terms."""

    name = 'tokenized_text'
    type = Terms
    kind = 'O'

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
        return SearchArray

    def __repr__(self):
        return 'TermsDtype()'

    @property
    def na_value(self):
        return Terms({})

    def valid_value(self, value):
        return isinstance(value, dict) or pd.isna(value) or isinstance(value, Terms)


register_extension_dtype(TermsDtype)


def ws_tokenizer(string):
    if pd.isna(string):
        return []
    if not isinstance(string, str):
        raise ValueError("Expected a string")
    return string.split()


def _row_to_postings_row(doc_id, row, doc_len, term_dict, posns: PosnBitArray):
    tfs = {}
    labeled_posns = {}
    for term_idx in row.cols:
        term = term_dict.get_term(term_idx)
        tfs[term] = 1
        enc_term_posns = posns.doc_encoded_posns(term_idx, doc_id=doc_id)
        labeled_posns[term] = enc_term_posns

    result = Terms(tfs, posns=labeled_posns,
                   doc_len=doc_len, encoded=True)
    return result


class SearchArray(ExtensionArray):
    """An array of tokenized text (Terms).

    Use the class method `index` to create a SearchArray from a list of strings and tokenizer

    """

    dtype = TermsDtype()

    def __init__(self, postings, tokenizer=ws_tokenizer, avoid_copies=True):
        # Check dtype, raise TypeError
        if not is_list_like(postings):
            raise TypeError("Expected list-like object, got {}".format(type(postings)))

        self.avoid_copies = avoid_copies
        self.tokenizer = tokenizer
        self.term_mat, self.posns, \
            self.term_dict, self.avg_doc_length, \
            self.doc_lens = build_index_from_terms_list(postings, Terms)
        self.corpus_size = len(self.doc_lens)

    @classmethod
    def index(cls, array: Iterable,
              tokenizer=ws_tokenizer,
              truncate=False,
              batch_size=100000,
              avoid_copies=True,
              workers=4,
              cache_gt_than=25,
              data_dir: Optional[str] = None,
              autowarm=True) -> 'SearchArray':
        """Index an array of strings into SearchArray using tokenizer.

        Parameters
        ----------
        array : Iterable
            A list of strings to index
        tokenizer : Callable
            A function that takes a string and returns an iterable of tokens
        truncate : bool
            If true, truncate the tokenized strings to a maximum length (2^18-1)
        batch_size : int
            For tuning memory usage, the number of strings to encode at once.
            (which will be merged into one index at the end). Higher the more memory
            will be used.
        avoid_copies : bool
            If true, avoid copying the term dictionary and positions during normal
            pandas copy operations
        autowarm : bool
            If true, precompute docfreq / term freqs for most common terms
            after indexing
        """
        if not is_list_like(array):
            raise TypeError("Expected list-like object, got {}".format(type(array)))

        term_mat, posns, term_dict, avg_doc_length, doc_lens =\
            build_index_from_tokenizer(array, tokenizer, batch_size=batch_size,
                                       truncate=truncate,
                                       data_dir=data_dir,
                                       cache_gt_than=cache_gt_than,
                                       workers=workers)

        if autowarm:
            posns.warm()

        postings = cls([], tokenizer=tokenizer, avoid_copies=avoid_copies)
        postings.term_mat = term_mat
        postings.posns = posns
        postings.term_dict = term_dict
        postings.avg_doc_length = avg_doc_length
        postings.doc_lens = doc_lens
        postings.corpus_size = len(doc_lens)
        return postings

    def warm(self):
        self.posns.warm()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new SearchArray from a sequence of scalars (PostingRow or convertible into)."""
        if dtype is not None:
            if not isinstance(dtype, TermsDtype):
                return scalars
        if isinstance(scalars, np.ndarray) and scalars.dtype == TermsDtype():
            return cls(scalars)
        # String types
        elif isinstance(scalars, np.ndarray) and scalars.dtype.kind in 'US':
            return cls(scalars)
        # Other objects
        elif isinstance(scalars, np.ndarray) and scalars.dtype != object:
            return scalars
        return cls(scalars)

    def memory_usage(self, deep=False):
        """Return memory usage of this array in bytes."""
        return self.nbytes

    @property
    def nbytes(self):
        return self.term_mat.nbytes + self.posns.nbytes + self.doc_lens.nbytes + self.term_dict.nbytes

    def __getitem__(self, key):
        key = pd.api.indexers.check_array_indexer(self, key)
        # Want to take rows of term freqs
        if isinstance(key, numbers.Integral):
            try:
                rows = self.term_mat[key]
                doc_len = self.doc_lens[key]
                doc_id = key
                if doc_id < 0:
                    doc_id += len(self)
                return _row_to_postings_row(doc_id, rows[0], doc_len,
                                            self.term_dict, self.posns)
            except IndexError:
                raise IndexError("index out of bounds")
        else:
            # Construct a sliced view of this array
            sliced_tfs = self.term_mat.slice(key)
            if not self.avoid_copies:
                sliced_posns = self.posns.slice(sliced_tfs.rows) if not self.avoid_copies else self.posns
            else:
                sliced_posns = self.posns.filter(sliced_tfs.rows)

            arr = SearchArray([], tokenizer=self.tokenizer)
            arr.term_mat = sliced_tfs
            arr.doc_lens = self.doc_lens[key]
            arr.posns = sliced_posns
            arr.term_dict = self.term_dict
            arr.avg_doc_length = self.avg_doc_length
            arr.corpus_size = self.corpus_size
            return arr

    def __setitem__(self, key, value):
        """Set an item in the array."""
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, SearchArray):
            value = value.to_numpy()
        if isinstance(value, list):
            value = np.asarray(value, dtype=object)

        if not isinstance(value, np.ndarray) and not self.dtype.valid_value(value):
            raise ValueError(f"Cannot set non-object array to SearchArray -- you passed type:{type(value)} -- {value}")

        # Cant set a single value to an array
        if isinstance(key, numbers.Integral) and isinstance(value, np.ndarray):
            raise ValueError("Cannot set a single value to an array")

        try:
            is_encoded = False
            posns = None
            term_mat = np.asarray([])
            doc_lens = np.asarray([])
            if isinstance(value, float):
                term_mat = np.asarray([value])
                doc_lens = np.asarray([0])
            elif isinstance(value, Terms):
                term_mat = np.asarray([value.tf_to_dense(self.term_dict)])
                doc_lens = np.asarray([value.doc_len])
                is_encoded = value.encoded
                posns = [value.raw_positions(self.term_dict)]
            elif isinstance(value, np.ndarray):
                term_mat = np.asarray([x.tf_to_dense(self.term_dict) for x in value])
                doc_lens = np.asarray([x.doc_len for x in value])
                is_encoded = value[0].encoded if len(value) > 0 else False
                posns = [x.raw_positions(self.term_dict) for x in value]
            np.nan_to_num(term_mat, copy=False, nan=0)
            self.term_mat[key] = term_mat
            self.doc_lens[key] = doc_lens

            if posns is not None:
                self.posns.insert(key, posns, is_encoded)

            # Assume we have a positions for each term, doc pair. We can just update it.
            # Otherwise we would have added new terms
        except TermMissingError:
            self._add_new_terms(key, value)

    def _add_new_terms(self, key, value):
        msg = """Adding new terms! This might not be good if you tokenized this new text
                 with a different tokenizer.

                 Also. This is slow."""
        warnings.warn(msg)

        scan_value = value
        if isinstance(value, Terms):
            scan_value = np.asarray([value])
        for row in scan_value:
            for term in row.terms():
                self.term_dict.add_term(term[0])

        self.term_mat.resize((self.term_mat.shape[0], len(self.term_dict)))
        # Ensure posns_lookup has at least max self.posns
        self[key] = value

    def value_counts(
        self,
        dropna: bool = True,
    ):
        if dropna:
            counts = Counter(self[:])
            counts.pop(Terms({}), None)
        else:
            counts = Counter(self[:])
        return pd.Series(counts)

    def __len__(self):
        len_rval = len(self.term_mat.rows)
        return len_rval

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
        if isinstance(other, SearchArray):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            else:
                # Compatible term dicts, and same term freqs
                # (not looking at positions, maybe we should?)
                if self.term_dict.compatible(other.term_dict):
                    return (self.term_mat == other.term_mat) & (self.doc_lens == other.doc_lens)
                else:
                    return np.zeros(len(self), dtype=bool)
            # return np.array(self[:]) == np.array(other[:])

        # When other is a scalar value
        elif isinstance(other, Terms):
            other = SearchArray([other], tokenizer=self.tokenizer)
            warnings.warn("Comparing a scalar value to a SearchArray. This is slow.")
            return np.array(self[:]) == np.array(other[:])

        # When other is a sequence but not an ExtensionArray
        # its an array of dicts
        elif is_list_like(other):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            # We actually don't know how it was tokenized
            other = SearchArray(other, tokenizer=self.tokenizer)
            return np.array(self[:]) == np.array(other[:])

        # Return False where 'other' is neither the same length nor a scalar
        else:
            return np.full(len(self), False)

    def isna(self):
        # Every row with all 0s
        empties = self.doc_lens == 0
        return empties

    def unique(self):
        """This is a hack for colab which wants to visit every element of the array."""
        logger.warning("Unique called on SearchArray. This is not supported.")
        return self[:]

    def __iter__(self):
        if len(self) > 10000:
            warning_text = """Iterating over SearchArray is very slow and not reccomended.

            If you're looping a dataframe, slice out the non-SearchArray columns first."""
            logger.warning(warning_text)
            warnings.warn(warning_text)
        return super().__iter__()

    def take(self, indices, allow_fill=False, fill_value=None):
        # Want to take rows of term freqs
        row_indices = np.arange(len(self.term_mat.rows))
        # Take within the row indices themselves
        result_indices = take(row_indices, indices, allow_fill=allow_fill, fill_value=-1)

        if allow_fill and -1 in result_indices:
            if fill_value is None or pd.isna(fill_value):
                fill_value = Terms({}, encoded=True)

            to_fill_mask = result_indices == -1
            # This is slow as it rebuilds all the term dictionaries
            # on the subsequent assignment lines
            # However, this case tends to be the exception for
            # most dataframe operations
            taken = SearchArray([fill_value] * len(result_indices))
            taken[~to_fill_mask] = self[result_indices[~to_fill_mask]].copy()

            return taken
        else:
            taken = self[result_indices].copy()
            return taken

    def copy(self):
        postings_arr = SearchArray([], tokenizer=self.tokenizer)
        postings_arr.doc_lens = self.doc_lens.copy()
        postings_arr.term_mat = self.term_mat.copy()
        postings_arr.posns = self.posns
        postings_arr.term_dict = self.term_dict
        postings_arr.avg_doc_length = self.avg_doc_length
        postings_arr.corpus_size = self.corpus_size

        if not self.avoid_copies:
            postings_arr.posns = self.posns.copy()
            postings_arr.term_dict = self.term_dict.copy()
        return postings_arr

    @classmethod
    def _concat_same_type(cls, to_concat):
        concatenated_data = np.concatenate([ea[:] for ea in to_concat])
        return SearchArray(concatenated_data, tokenizer=to_concat[0].tokenizer)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def _values_for_factorize(self):
        """Return an array and missing value suitable for factorization (ie grouping)."""
        arr = np.asarray(self[:], dtype=object)
        return arr, Terms({})

    def _check_token_arg(self, token):
        if isinstance(token, str):
            return token
        elif isinstance(token, list) and len(token) == 1:
            return token[0]
        elif isinstance(token, list):
            return token
        else:
            raise TypeError("Expected a string or list of strings for phrases")

    def memory_report(self, N=1000):
        """Return a string with memory usage information."""
        term_mat_bytes = self.term_mat.nbytes
        posns_bytes = self.posns.nbytes
        term_dict_bytes = self.term_dict.nbytes
        num_terms = len(self.term_dict)
        top_10_terms_size = []

        for i in range(N):
            term_len = len(self.posns.encoded_term_posns[i])
            term = self.term_dict.get_term(i)
            term_bytes = term_len * 8
            top_10_terms_size.append((term, term_bytes))

        top_10_terms_size = sorted(top_10_terms_size, key=lambda x: x[1], reverse=True)

        report = f"""
        SearchArray Memory Report
        -------------------------
        Number of Terms: {num_terms}
        -------------------------
        Term Matrix:     {bytes_to_human_readable(term_mat_bytes)}
        Positions:       {bytes_to_human_readable(posns_bytes)}
        Term Dictionary: {bytes_to_human_readable(term_dict_bytes)}

        """

        cum_sum = 0
        for i in range(N):
            term, term_bytes = top_10_terms_size[i]
            cum_sum += term_bytes
            report += f"        Term {i}: {term} - {bytes_to_human_readable(term_bytes)} - Cumulative: {bytes_to_human_readable(cum_sum)}\n"
        return report

    # ***********************************************************
    # Search API
    # ***********************************************************
    def termfreqs(self, token: Union[List[str], str],
                  slop: int = 0,
                  min_posn: Optional[int] = None,
                  max_posn: Optional[int] = None) -> np.ndarray:
        token = self._check_token_arg(token)
        if isinstance(token, list):
            return self._phrase_freq(token, slop=slop, min_posn=min_posn, max_posn=max_posn)

        try:
            term_id = self.term_dict.get_term_id(token)
            slice_of_rows = self.term_mat.rows
            if self.term_mat.subset:
                matches = np.zeros(len(self), dtype=np.float32)
                doc_ids, termfreqs = self.posns.termfreqs(term_id,
                                                          doc_ids=slice_of_rows,
                                                          min_posn=min_posn,
                                                          max_posn=max_posn)
                mask = np.isin(self.term_mat.rows, doc_ids)
                matches[mask] = termfreqs
                return matches
            else:
                doc_ids, termfreqs = self.posns.termfreqs(term_id,
                                                          doc_ids=None,
                                                          min_posn=min_posn,
                                                          max_posn=max_posn)
                return as_dense(doc_ids, termfreqs, len(self))
        except TermMissingError:
            return np.zeros(len(self), dtype=np.float32)

    def docfreq(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError("Expected a string")
        # Count number of rows where the term appears
        try:
            return self.posns.docfreq(self.term_dict.get_term_id(token))
        except TermMissingError:
            return 0

    def doclengths(self) -> np.ndarray:
        return self.doc_lens

    def score(self, token: Union[str, List[str]],
              similarity: Similarity = default_bm25,
              slop: int = 0,
              min_posn: Optional[int] = None,
              max_posn: Optional[int] = None) -> np.ndarray:
        """Score each doc using a similarity function.

        Parameters
        ----------
        token : str or list of str of what to search (already tokenized)
        similarity : How to score the documents. Default is BM25.
        min_posn : int - minimum position of the term in the document, in multiples of 18
        max_posn : int - maximum position of the term in the document, in multiples of 18
        """
        # Get term freqs per token
        token = self._check_token_arg(token)

        # For expensive toknes, we compute doc freq first, so we
        # cache them in the DF cache, to let TF cache know it should be cached
        tokens_l = [token] if isinstance(token, str) else token
        all_dfs = np.asarray([self.docfreq(token) for token in tokens_l])

        tfs = self.termfreqs(token, min_posn=min_posn, max_posn=max_posn,
                             slop=slop)
        token = self._check_token_arg(token)
        doc_lens = self.doclengths()

        scores = similarity(tfs, all_dfs, doc_lens, self.avg_doc_length, self.corpus_size)
        return scores

    def positions(self, token: str, key=None) -> List[np.ndarray]:
        """Return a list of lists of positions of the given term."""
        term_id = self.term_dict.get_term_id(token)
        key = self.term_mat.rows[key] if key is not None else self.term_mat.rows
        posns = self.posns.positions(term_id, doc_ids=key)
        return posns

    def _phrase_freq(self, tokens: List[str],
                     slop=0,
                     min_posn: Optional[int] = None,
                     max_posn: Optional[int] = None) -> np.ndarray:
        if slop > 0:
            logger.warning("!! Slop is experimental and may be slow, crash, or inaccurate etc")
        try:
            # Decide how/if we need to filter doc ids
            term_ids = [self.term_dict.get_term_id(token) for token in tokens]
            phrase_freqs = self.posns.phrase_freqs(term_ids,
                                                   slop=slop,
                                                   min_posn=min_posn,
                                                   max_posn=max_posn)
            if self.term_mat.subset:
                return phrase_freqs[self.term_mat.rows]
            return phrase_freqs
        except TermMissingError:
            if self.term_mat.subset:
                return np.zeros(len(self), dtype=np.float32)
            return self.posns.empty_buffer()
