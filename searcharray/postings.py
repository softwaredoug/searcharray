"""Tokenized, searchable text as a pandas dtype."""
import pandas as pd
import numbers
from collections import Counter, defaultdict
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype
from pandas.api.types import is_list_like
from pandas.api.extensions import take
import json
import warnings
import logging
from time import perf_counter


import numpy as np
from searcharray.utils.row_viewable_matrix import RowViewableMatrix
from searcharray.term_dict import TermDict, TermMissingError
from searcharray.phrase.scan_merge import scan_merge_ins
from searcharray.phrase.posn_diffs import compute_phrase_freqs
from searcharray.phrase.middle_out import PosnBitArrayBuilder, PosnBitArrayAlreadyEncBuilder, PosnBitArray
from searcharray.utils.mat_set import SparseMatSetBuilder

logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


class PostingsRow:
    """Wrapper around a row of a postings matrix.

    We can't easily directly use a dictionary as a cell type in pandas.
    See:

    https://github.com/pandas-dev/pandas/issues/17777
    """

    def __init__(self,
                 postings,
                 doc_len: int = 0,
                 posns: dict = None,
                 encoded=False):
        self.postings = postings
        self.posns = None
        self.encoded = encoded
        self.doc_len = doc_len

        if posns is not None:
            for term, term_posns in posns.items():
                if not isinstance(term_posns, np.ndarray):
                    posns[term] = np.array(term_posns)
                    if len(posns[term].shape) != 1:
                        raise ValueError("Positions must be a 1D array.")
        self.posns = posns
        self._validate_posns()

    def _validate_posns(self):
        # Confirm every term in positions also in postings
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
            posns = np.array(self.posns[term])
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
        if self.encoded:
            rval = f"PostingsRow({repr(self.postings)}, encoded=True)"
        else:
            rval = f"PostingsRow({repr(self.postings)})"
        return rval

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        # Flip to the other implementation if we're comparing to a PostingsArray
        # to get a boolean array back
        if isinstance(other, PostingsArray):
            return other == self
        same_postings = isinstance(other, PostingsRow) and self.postings == other.postings
        if same_postings and self.doc_len == other.doc_len:
            return True

    def __lt__(self, other):
        # return isinstance(other, PostingsRow) and hash(self) < hash(other)
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


class PostingsDtype(ExtensionDtype):
    name = 'tokenized_text'
    type = PostingsRow
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
        return PostingsRow({})

    def valid_value(self, value):
        return isinstance(value, dict) or pd.isna(value) or isinstance(value, PostingsRow)


register_extension_dtype(PostingsDtype)


def ws_tokenizer(string):
    if pd.isna(string):
        return []
    if not isinstance(string, str):
        raise ValueError("Expected a string")
    return string.split()

# To add positions
# Row/Col -> index to roaring bitmap
# Must be slicable


def _build_index_from_dict(postings):
    """Bulid an index from postings that are already tokenized and point at their term frequencies."""
    start = perf_counter()
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()
    doc_lens = []
    avg_doc_length = 0
    num_postings = 0
    add_term_time = 0
    set_time = 0
    get_posns_time = 0
    set_posns_time = 0
    posns = PosnBitArrayBuilder()
    posns_enc = PosnBitArrayAlreadyEncBuilder()

    # COPY 1
    # Consume generator (tokenized postings) into list
    # its faster this way?
    postings = list(postings)
    logger.debug(f"Tokenized {len(postings)} documents in {perf_counter() - start} seconds")

    # COPY 2
    # Build dict for sparse matrix
    # this is faster that directly using the matrix
    # https://www.austintripp.ca/blog/2018/09/12/sparse-matrices-tips1
    for doc_id, tokenized in enumerate(postings):
        if isinstance(tokenized, dict):
            tokenized = PostingsRow(tokenized, doc_len=len(tokenized))
        elif not isinstance(tokenized, PostingsRow):
            raise TypeError("Expected a PostingsRow or a dict")

        if tokenized.encoded:
            posns = posns_enc

        doc_lens.append(tokenized.doc_len)
        avg_doc_length += doc_lens[-1]
        terms = []
        for token, term_freq in tokenized.terms():
            add_term_start = perf_counter()
            term_id = term_dict.add_term(token)
            add_term_time += perf_counter() - add_term_start

            set_time_start = perf_counter()
            terms.append(term_id)
            set_time += perf_counter() - set_time_start

            get_posns_start = perf_counter()
            positions = tokenized.positions(token)
            get_posns_time += perf_counter() - get_posns_start

            set_posns_start = perf_counter()
            if positions is not None:
                posns.add_posns(doc_id, term_id, positions)

            set_posns_time += perf_counter() - set_posns_start

        set_time_start = perf_counter()
        term_doc.append(terms)
        set_time += perf_counter() - set_time_start

        if doc_id % 1000 == 0:
            logger.debug(f"Indexed {doc_id} documents in {perf_counter() - start} seconds")
            logger.debug(f"   add time: {add_term_time}")
            logger.debug(f"   set time: {set_time}")
            logger.debug(f"   get posns time: {get_posns_time}")
            logger.debug(f"   set posns time: {set_posns_time}")
        posns.ensure_capacity(doc_id)
        num_postings += 1

    if num_postings > 0:
        avg_doc_length /= num_postings

    logger.debug(f"Indexed {num_postings} documents in {perf_counter() - start} seconds")

    bit_posns = posns.build()
    logger.info(f"Bitwis Posn memory usage: {bit_posns.nbytes / 1024 / 1024} MB")

    return RowViewableMatrix(term_doc.build()), bit_posns, term_dict, avg_doc_length, np.array(doc_lens)


def _row_to_postings_row(doc_id, row, doc_len, term_dict, posns: PosnBitArray):
    tfs = {}
    labeled_posns = {}
    for term_idx in row.cols:
        term = term_dict.get_term(term_idx)
        tfs[term] = 1
        enc_term_posns = posns.doc_encoded_posns(term_idx, doc_id=doc_id)
        labeled_posns[term] = enc_term_posns

    result = PostingsRow(tfs, posns=labeled_posns,
                         doc_len=doc_len, encoded=True)
    # TODO add positions
    return result


# Logically a PostingsArray is a document represented as follows:
#
#   docs = [
#       {"foo": 1, "bar": 2, "baz": 1}, # doc 0 term->freq
#       {"foo": 2, "bar": 4, "baz": 8, "the"}, # doc 0 term->freq
#       ...
#   ]
#
# This postings will build its own term_dict and term_freqs
class PostingsArray(ExtensionArray):
    dtype = PostingsDtype()

    def __init__(self, postings, tokenizer=ws_tokenizer):
        # Check dtype, raise TypeError
        if not is_list_like(postings):
            raise TypeError("Expected list-like object, got {}".format(type(postings)))

        self.tokenizer = tokenizer
        self.term_freqs, self.posns, \
            self.term_dict, self.avg_doc_length, \
            self.doc_lens = _build_index_from_dict(postings)

    @classmethod
    def index(cls, array, tokenizer=ws_tokenizer):
        """Index an array of strings using tokenizer."""
        # Convert strings to expected scalars (dict -> term freqs)
        if not is_list_like(array):
            raise TypeError("Expected list-like object, got {}".format(type(array)))
        if not all(isinstance(x, str) or pd.isna(x) for x in array):
            raise TypeError("Expected a list of strings to tokenize")

        def tokenized_docs(docs):
            for doc_id, doc in enumerate(docs):
                if pd.isna(doc):
                    yield PostingsRow({})
                else:
                    token_stream = tokenizer(doc)
                    term_freqs = Counter(token_stream)
                    positions = defaultdict(list)
                    for posn in range(len(token_stream)):
                        positions[token_stream[posn]].append(posn)
                    yield PostingsRow(term_freqs,
                                      doc_len=len(token_stream),
                                      posns=positions)

        return cls(tokenized_docs(array), tokenizer)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new PostingsArray from a sequence of scalars (PostingRow or convertible into)."""
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
        return self.term_freqs.nbytes + self.posns.nbytes + self.doc_lens.nbytes + self.term_dict.nbytes

    def __getitem__(self, key):
        key = pd.api.indexers.check_array_indexer(self, key)
        # Want to take rows of term freqs
        if isinstance(key, numbers.Integral):
            try:
                rows = self.term_freqs[key]
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
            sliced_tfs = self.term_freqs.slice(key)
            sliced_posns = self.posns.slice(key)
            arr = PostingsArray([], tokenizer=self.tokenizer)
            arr.term_freqs = sliced_tfs
            arr.doc_lens = self.doc_lens[key]
            arr.posns = sliced_posns
            arr.term_dict = self.term_dict
            arr.avg_doc_length = self.avg_doc_length
            return arr

    def __setitem__(self, key, value):
        """Set an item in the array."""
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, PostingsArray):
            value = value.to_numpy()
        if isinstance(value, list):
            value = np.asarray(value, dtype=object)

        if not isinstance(value, np.ndarray) and not self.dtype.valid_value(value):
            raise ValueError(f"Cannot set non-object array to PostingsArray -- you passed type:{type(value)} -- {value}")

        # Cant set a single value to an array
        if isinstance(key, numbers.Integral) and isinstance(value, np.ndarray):
            raise ValueError("Cannot set a single value to an array")

        try:
            is_encoded = False
            posns = None
            term_freqs = np.asarray([])
            doc_lens = np.asarray([])
            if isinstance(value, float):
                term_freqs = np.asarray([value])
                doc_lens = np.asarray([0])
            elif isinstance(value, PostingsRow):
                term_freqs = np.asarray([value.tf_to_dense(self.term_dict)])
                doc_lens = np.asarray([value.doc_len])
                is_encoded = value.encoded
                posns = [value.raw_positions(self.term_dict)]
            elif isinstance(value, np.ndarray):
                term_freqs = np.asarray([x.tf_to_dense(self.term_dict) for x in value])
                doc_lens = np.asarray([x.doc_len for x in value])
                is_encoded = value[0].encoded if len(value) > 0 else False
                posns = [x.raw_positions(self.term_dict) for x in value]
            np.nan_to_num(term_freqs, copy=False, nan=0)
            self.term_freqs[key] = term_freqs
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
        if isinstance(value, PostingsRow):
            scan_value = np.asarray([value])
        for row in scan_value:
            for term in row.terms():
                self.term_dict.add_term(term[0])

        self.term_freqs.resize((self.term_freqs.shape[0], len(self.term_dict)))
        # Ensure posns_lookup has at least max self.posns
        self[key] = value

    def value_counts(
        self,
        dropna: bool = True,
    ):
        if dropna:
            counts = Counter(self[:])
            counts.pop(PostingsRow({}), None)
        else:
            counts = Counter(self[:])
        return pd.Series(counts)

    def __len__(self):
        len_rval = len(self.term_freqs.rows)
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
        if isinstance(other, PostingsArray):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            else:
                # Compatible term dicts, and same term freqs
                # (not looking at positions, maybe we should?)
                if self.term_dict.compatible(other.term_dict):
                    return (self.term_freqs == other.term_freqs) & (self.doc_lens == other.doc_lens)
                else:
                    return np.zeros(len(self), dtype=bool)
            # return np.array(self[:]) == np.array(other[:])

        # When other is a scalar value
        elif isinstance(other, PostingsRow):
            other = PostingsArray([other], tokenizer=self.tokenizer)
            warnings.warn("Comparing a scalar value to a PostingsArray. This is slow.")
            return np.array(self[:]) == np.array(other[:])

        # When other is a sequence but not an ExtensionArray
        # its an array of dicts
        elif is_list_like(other):
            if len(self) != len(other):
                return False
            elif len(other) == 0:
                return np.array([], dtype=bool)
            # We actually don't know how it was tokenized
            other = PostingsArray(other, tokenizer=self.tokenizer)
            return np.array(self[:]) == np.array(other[:])

        # Return False where 'other' is neither the same length nor a scalar
        else:
            return np.full(len(self), False)

    def isna(self):
        # Every row with all 0s
        empties = self.doc_lens == 0
        return empties

    def take(self, indices, allow_fill=False, fill_value=None):
        # Want to take rows of term freqs
        row_indices = np.arange(len(self.term_freqs.rows))
        # Take within the row indices themselves
        result_indices = take(row_indices, indices, allow_fill=allow_fill, fill_value=-1)

        if allow_fill and -1 in result_indices:
            if fill_value is None or pd.isna(fill_value):
                fill_value = PostingsRow({}, encoded=True)

            to_fill_mask = result_indices == -1
            # This is slow as it rebuilds all the term dictionaries
            # on the subsequent assignment lines
            # However, this case tends to be the exception for
            # most dataframe operations
            taken = PostingsArray([fill_value] * len(result_indices))
            taken[~to_fill_mask] = self[result_indices[~to_fill_mask]].copy()

            return taken
        else:
            taken = self[result_indices].copy()
            return taken

    def copy(self):
        postings_arr = PostingsArray([], tokenizer=self.tokenizer)
        postings_arr.doc_lens = self.doc_lens.copy()
        postings_arr.posns = self.posns.copy()
        postings_arr.term_freqs = self.term_freqs.copy()
        postings_arr.term_dict = self.term_dict.copy()
        postings_arr.avg_doc_length = self.avg_doc_length
        return postings_arr

    @classmethod
    def _concat_same_type(cls, to_concat):
        concatenated_data = np.concatenate([ea[:] for ea in to_concat])
        return PostingsArray(concatenated_data, tokenizer=to_concat[0].tokenizer)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def _values_for_factorize(self):
        """Return an array and missing value suitable for factorization (ie grouping)."""
        arr = np.asarray(self[:], dtype=object)
        return arr, PostingsRow({})

    # One way to stack
    #  np.array_split(posns_mat[[1,2]].indices, posns_mat[[1,2]].indptr)

    def _check_token_arg(self, token):
        if isinstance(token, str):
            return token
        elif isinstance(token, list) and len(token) == 1:
            return token[0]
        elif isinstance(token, list):
            return token
        else:
            raise TypeError("Expected a string or list of strings for phrases")

    # ***********************************************************
    # Naive implementations of search functions to clean up later
    # ***********************************************************
    def term_freq(self, token):
        token = self._check_token_arg(token)
        if isinstance(token, list):
            return self.phrase_freq(token)

        try:
            term_id = self.term_dict.get_term_id(token)
            matches = np.zeros(len(self), dtype=int)
            doc_ids, termfreqs = self.posns.termfreqs(term_id)
            doc_ids_in_matches = np.intersect1d(self.term_freqs.rows, doc_ids).astype(np.uint32)
            mask = np.isin(self.term_freqs.rows, doc_ids_in_matches)
            matches[mask] = termfreqs
            return matches
        except TermMissingError:
            return np.zeros(len(self), dtype=int)

    def doc_freq(self, token):
        if not isinstance(token, str):
            raise TypeError("Expected a string")
        # Count number of rows where the term appears
        term_freq = self.term_freq(token)
        return np.sum(term_freq > 0)

    def doc_lengths(self):
        return self.doc_lens

    def match(self, token, slop=1):
        """Return a boolean numpy array indicating which elements contain the given term."""
        token = self._check_token_arg(token)
        if isinstance(token, list):
            term_freq = self.phrase_freq(token)
        else:
            term_freq = self.term_freq(token)
        return term_freq > 0

    def bm25_idf(self, token, doc_stats=None):
        """Calculate the (Lucene) idf for a term.

        idf, computed as log(1 + (N - n + 0.5) / (n + 0.5))
        """
        token = self._check_token_arg(token)
        if isinstance(token, list):
            return self.bm25_phrase_idf(token)

        df = self.doc_freq(token)
        num_docs = len(self)
        return np.log(1 + (num_docs - df + 0.5) / (df + 0.5))

    def bm25_phrase_idf(self, tokens):
        """Calculate the idf for a phrase.

        This is the sum of the idfs of the individual terms.
        """
        idfs = [self.bm25_idf(term) for term in tokens]
        return np.sum(idfs)

    def bm25_tf(self, token, k1=1.2, b=0.75, slop=1):
        """Calculate the (Lucene) BM25 tf for a term.

        tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl))
        """
        tf = self.term_freq(token)
        score = tf / (tf + k1 * (1 - b + b * self.doc_lengths() / self.avg_doc_length))
        return score

    def bm25(self, token, doc_stats=None, k1=1.2, b=0.75):
        """Score each doc using BM25.

        Parameters
        ----------
        token : str or list of str of what to search (already tokenized)
        doc_stats : tuple of doc stats to use (avg_doc_length, num_docs, doc_count). Defaults to index stats.
        k1 : float, optional BM25 param. Defaults to 1.2.
        b : float, optional BM25 param. Defaults to 0.75.
        """
        # Get term freqs per token
        token = self._check_token_arg(token)
        return self.bm25_idf(token, doc_stats=doc_stats) * self.bm25_tf(token)

    def positions(self, token, key=None):
        """Return a list of lists of positions of the given term."""
        term_id = self.term_dict.get_term_id(token)
        posns = self.posns.positions(term_id, key=key)
        return posns

    def and_query(self, tokens):
        """Return a mask on the postings array indicating which elements contain all terms."""
        masks = [self.match(term) for term in tokens]
        mask = np.ones(len(self), dtype=bool)
        for curr_mask in masks:
            mask = mask & curr_mask
        return mask

    def phrase_freq(self, tokens, slop=1):
        return self.phrase_freq_every_diff(tokens, slop=slop)

    def phrase_freq_scan(self, tokens, mask=None, slop=1):
        if mask is None:
            mask = self.and_query(tokens)

        if np.sum(mask) == 0:
            return mask

        # Gather positions
        posns = [self.positions(token, mask) for token in tokens]
        phrase_freqs = np.zeros(len(self))

        phrase_freqs[mask] = scan_merge_ins(posns, phrase_freqs[mask], slop=slop)
        return phrase_freqs

    def phrase_freq_every_diff(self, tokens, slop=1):
        """Batch up calls to _phrase_freq_every_diff."""
        phrase_freqs = -np.ones(len(self))

        mask = self.and_query(tokens)
        phrase_freqs[~mask] = 0
        if np.sum(mask) == 0:
            return phrase_freqs

        term_posns = [self.positions(term, mask) for term in tokens]
        for width in [10, 20, 30, 40]:
            phrase_freqs[mask] = compute_phrase_freqs(term_posns,
                                                      phrase_freqs[mask],
                                                      slop=slop,
                                                      width=width)

        remaining_mask = phrase_freqs == -1
        if np.any(remaining_mask):
            remainder_freqs = self.phrase_freq_scan(tokens, mask=remaining_mask, slop=slop)
            phrase_freqs[remaining_mask] = remainder_freqs[remaining_mask]
        return phrase_freqs
