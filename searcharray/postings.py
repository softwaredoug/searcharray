"""Tokenized, searchable text as a pandas dtype."""
import pandas as pd
import numbers
from collections import Counter, defaultdict
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype
from pandas.api.types import is_list_like
from pandas.api.extensions import take
import json
import warnings

import numpy as np
from searcharray.term_dict import TermDict, TermMissingError

# Doc,Term -> freq
# Note scipy sparse switching to *_array, which is more numpy like
# However, as of now, these don't seem fully baked
from scipy.sparse import lil_matrix, csr_matrix


class PostingsRow:
    """Wrapper around a row of a postings matrix.

    We can't easily directly use a dictionary as a cell type in pandas.
    See:

    https://github.com/pandas-dev/pandas/issues/17777
    """

    def __init__(self, postings, posns=None):
        self.postings = postings
        self.posns = None
        if self.posns is not None and len(self.postings) != len(self.posns):
            raise ValueError("Postings and positions must be the same length.")
        else:
            self.posns = posns
            self._validate_posns()

    def _validate_posns(self):
        # Confirm every term in positions also in postings
        if self.posns is None:
            return
        for term in self.posns:
            if term not in self.postings:
                raise ValueError(f"Term {term} in positions but not in postings. ")

    def termfreq(self, term):
        return self.postings[term]

    def terms(self):
        return self.postings.items()

    def positions(self, term=None):
        if self.posns is None:
            return {}
        if term is None:
            return self.posns.items()
        else:
            return self.posns[term]

    def tf_to_dense(self, term_dict):
        """Convert to a dense vector of term frequencies."""
        dense = np.zeros(len(term_dict))
        for term, freq in self.terms():
            dense[term_dict.get_term_id(term)] = freq
        return dense

    def __len__(self):
        return len(self.postings)

    def __repr__(self):
        rval = f"PostingsRow({repr(self.postings)}, {repr(self.posns)})"
        return rval

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        # Flip to the other implementation if we're comparing to a PostingsArray
        # to get a boolean array back
        if isinstance(other, PostingsArray):
            return other == self
        return isinstance(other, PostingsRow) and self.postings == other.postings

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


class RowViewableMatrix:
    """A slicable matrix that can return views without copying."""

    def __init__(self, csr_mat: csr_matrix, rows: np.ndarray = None):
        self.mat = csr_mat
        if rows is None:
            self.rows = np.arange(self.mat.shape[0])
        elif isinstance(rows, numbers.Integral):
            self.rows = np.array([rows])
        else:
            self.rows = rows

    def slice(self, keys):
        return RowViewableMatrix(self.mat, self.rows[keys])

    def __setitem__(self, keys, values):
        # Replace nan with 0
        actual_keys = self.rows[keys]
        if isinstance(actual_keys, numbers.Integral):
            self.mat[actual_keys] = values
        elif len(actual_keys) > 0:
            self.mat[actual_keys] = values

    def copy_row_at(self, row):
        return self.mat[self.rows[row]]

    def copy(self):
        return RowViewableMatrix(self.mat.copy(), self.rows.copy())

    def sum(self, axis=0):
        return self.mat[self.rows].sum(axis=axis)

    def copy_col_at(self, col):
        return self.mat[self.rows, col]

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self.copy_row_at(key)
        else:
            return self.slice(key)

    @property
    def nbytes(self):
        return self.mat.data.nbytes + \
            self.mat.indptr.nbytes + \
            self.mat.indices.nbytes + \
            self.rows.nbytes

    @property
    def shape(self):
        return (len(self.rows), self.mat.shape[1])

    def resize(self, shape):
        self.mat.resize(shape)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"RowViewableMatrix({repr(self.mat)}, {repr(self.rows)})"

    def __str__(self):
        return f"RowViewableMatrix({str(self.mat)}, {str(self.rows)})"

# To add positions
# Row/Col -> index to roaring bitmap
# Must be slicable


def _build_index_from_dict(tokenized_postings):
    """Bulid an index from postings that are already tokenized and point at their term frequencies."""
    from time import perf_counter
    start = perf_counter()
    print("Building index...")
    freqs_table = lil_matrix((len(tokenized_postings), 0), dtype=np.uint8)
    posns_table = lil_matrix((len(tokenized_postings), 0), dtype=np.uint32)
    term_dict = TermDict()
    avg_doc_length = 0
    posns_lookup = [np.array([])]  # 0th is empty / None due to using a sparse matrix to lookup into this
    for doc_id, tokenized in enumerate(tokenized_postings):
        avg_doc_length += len(tokenized)
        for token, term_freq in tokenized.terms():
            term_id = term_dict.add_term(token)
            if term_id >= freqs_table.shape[1]:
                freqs_table.resize((freqs_table.shape[0], term_id + 1))
                posns_table.resize((posns_table.shape[0], term_id + 1))
            freqs_table[doc_id, term_id] = term_freq

            positions = tokenized.positions(token)
            if positions is not None:
                idx = len(posns_lookup)
                posns_lookup.append(np.array(positions))
                posns_table[doc_id, term_id] = idx
        if doc_id % 10000 == 0:
            print(f"Doc {doc_id} done -- {perf_counter() - start} seconds")

    if len(tokenized_postings) > 0:
        avg_doc_length /= len(tokenized_postings)

    assert freqs_table.shape == posns_table.shape
    print(f"Built index in {perf_counter() - start} seconds")
    return RowViewableMatrix(csr_matrix(freqs_table)), RowViewableMatrix(csr_matrix(posns_table)), posns_lookup, term_dict, avg_doc_length


def _row_to_postings_row(row, term_dict, posns, posns_lookup):
    tfs = {}
    non_zeros = row.nonzero()
    labeled_posns = {}
    for row_idx, col_idx in zip(non_zeros[0], non_zeros[1]):
        term = term_dict.get_term(col_idx)
        tfs[term] = int(row[row_idx, col_idx])
        posn = posns[col_idx]
        term_posns = posns_lookup[posn]
        labeled_posns[term] = term_posns

    result = PostingsRow(tfs, labeled_posns)
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
        if not all(isinstance(x, PostingsRow) or isinstance(x, dict) or pd.isna(x) for x in postings):
            raise TypeError("Expected a list of PostingsRow or dicts")

        # Convert all to postings rows
        as_postings = [PostingsRow(x) if isinstance(x, dict) else x for x in postings]

        self.tokenizer = tokenizer
        self.term_freqs, self.posns, self.posns_lookup, \
            self.term_dict, self.avg_doc_length = _build_index_from_dict(as_postings)
        if self.posns is not None and len(self.posns) > 0 and self.posns.shape[1] > 0:
            max_lookup = self.posns.mat.max()
            if max_lookup > len(self.posns_lookup):
                self.posns_lookup = np.resize(self.posns_lookup, max_lookup + 1)

    @classmethod
    def index(cls, array, tokenizer=ws_tokenizer):
        """Index an array of strings using tokenizer."""
        # Convert strings to expected scalars (dict -> term freqs)
        if not is_list_like(array):
            raise TypeError("Expected list-like object, got {}".format(type(array)))
        if not all(isinstance(x, str) or pd.isna(x) for x in array):
            raise TypeError("Expected a list of strings to tokenize")

        def tokenized_docs(docs):
            for doc in docs:
                if pd.isna(doc):
                    yield PostingsRow({})
                else:
                    token_stream = tokenizer(doc)
                    term_freqs = Counter(token_stream)
                    positions = defaultdict(list)
                    for posn in range(len(token_stream)):
                        positions[token_stream[posn]].append(posn)
                    yield PostingsRow(term_freqs, positions)

        return cls([a for a in tokenized_docs(array)], tokenizer)

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
        return self.term_freqs.nbytes + self.posns.nbytes

    def __getitem__(self, key):
        key = pd.api.indexers.check_array_indexer(self, key)
        # Want to take rows of term freqs
        if isinstance(key, numbers.Integral):
            try:
                rows = self.term_freqs[key]
                posn_keys = self.posns[key].toarray().flatten()

                return _row_to_postings_row(rows[0], self.term_dict, posn_keys, self.posns_lookup)
            except IndexError:
                raise IndexError("index out of bounds")
        else:
            # Construct a sliced view of this array
            sliced_tfs = self.term_freqs.slice(key)
            sliced_posns = self.posns.slice(key)
            arr = PostingsArray([], tokenizer=self.tokenizer)
            arr.term_freqs = sliced_tfs
            arr.posns = sliced_posns
            arr.posns_lookup = self.posns_lookup
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
            posns = None
            term_freqs = np.asarray([])
            if isinstance(value, float):
                term_freqs = np.asarray([value])
            elif isinstance(value, PostingsRow):
                term_freqs = np.asarray([value.tf_to_dense(self.term_dict)])
                posns = np.asarray([value.positions()])
            elif isinstance(value, np.ndarray):
                term_freqs = np.asarray([x.tf_to_dense(self.term_dict) for x in value])
                posns = np.asarray([x.positions() for x in value])
            np.nan_to_num(term_freqs, copy=False, nan=0)
            self.term_freqs[key] = term_freqs

            if posns is not None:
                update_rows = self.posns[key]
                for update_row_idx, new_posns_row in enumerate(posns):
                    for term, positions in new_posns_row:
                        term_id = self.term_dict.get_term_id(term)
                        lookup_location = update_rows[update_row_idx][0, term_id]
                        self.posns_lookup[lookup_location] = positions

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
        self.posns.resize((self.term_freqs.shape[0], len(self.term_dict)))
        # Ensure posns_lookup has at least max self.posns
        max_lookup = self.posns.mat.max()
        if max_lookup > len(self.posns_lookup):
            self.posns_lookup = np.resize(self.posns_lookup, max_lookup + 1)
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
        return len(self.term_freqs.rows)

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
            return np.array(self[:]) == np.array(other[:])

        # When other is a scalar value
        elif isinstance(other, PostingsRow):
            other = PostingsArray([other], tokenizer=self.tokenizer)
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
        key_slice_all = slice(None)
        sliced = self.term_freqs.slice(key_slice_all)
        empties = np.asarray((sliced.sum(axis=1) == 0).flatten())[0]
        return empties

    def take(self, indices, allow_fill=False, fill_value=None):
        # Want to take rows of term freqs
        row_indices = np.arange(len(self.term_freqs.rows))
        # Take within the row indices themselves
        result_indices = take(row_indices, indices, allow_fill=allow_fill, fill_value=-1)

        if allow_fill and -1 in result_indices:
            if fill_value is None or pd.isna(fill_value):
                fill_value = PostingsRow({})

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
        postings_arr.posns = self.posns.copy()
        postings_arr.posns_lookup = self.posns_lookup.copy()
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

    # ***********************************************************
    # Naive implementations of search functions to clean up later
    # ***********************************************************
    def term_freq(self, tokenized_term):
        if not isinstance(tokenized_term, str):
            raise TypeError("Expected a string")

        try:
            term_id = self.term_dict.get_term_id(tokenized_term)
            matches = self.term_freqs.copy_col_at(term_id).todense().flatten()
            matches = np.asarray(matches).flatten()
            return matches
        except TermMissingError:
            return np.zeros(len(self), dtype=int)

    def doc_freq(self, tokenized_term):
        if not isinstance(tokenized_term, str):
            raise TypeError("Expected a string")
        # Count number of rows where the term appears
        term_freq = self.term_freq(tokenized_term)
        return np.sum(term_freq > 0)

    def doc_lengths(self):
        return np.array(self.term_freqs.sum(axis=1).flatten())[0]

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
        numer = (k1 + 1) * tf
        denom = k1 * (1 - b + b * (self.doc_lengths() / self.avg_doc_length))
        return numer / denom

    def bm25(self, tokenized_term, k1=1.2, b=0.75):
        """Score each doc using BM25."""
        return self.bm25_idf(tokenized_term) * self.bm25_tf(tokenized_term)

    def _posns_lookup_to_csr(self):
        """Convert the posns_lookup to a csr_matrix."""
        from scipy.sparse import csr_matrix
        # This is a list of lists of positions
        # We want to convert it to a csr_matrix
        # where each row is a document and each column is a position
        mat = csr_matrix((len(self.posns_lookup) + 1, 255), dtype=np.int8)
        for row in range(len(self.posns_lookup)):
            for col in self.posns_lookup[row]:
                if col > mat.shape[1]:
                    mat.resize((mat.shape[0], col + 1))
                print(row)
                mat[row, col] = 1
        return mat

    def positions(self, tokenized_term, key=None):
        """Return a list of lists of positions of the given term."""
        term_id = self.term_dict.get_term_id(tokenized_term)

        if key is not None:
            posns_to_lookup = self.posns[key].copy_col_at(term_id)
        else:
            posns_to_lookup = self.posns.copy_col_at(term_id)

        # This could be faster if posns_lookup was more row slicable
        posns_to_lookup = posns_to_lookup.toarray().flatten()
        posns = [self.posns_lookup[lookup] for lookup in posns_to_lookup]
        # posns_mat = self._posns_lookup_to_csr()
        # this_mat = posns_mat[posns_to_lookup]
        # nonzeros = this_mat.nonzero()
        # this_mat[nonzeros] = (nonzeros[1] + 1)
        return posns

    def and_query(self, tokenized_terms):
        """Return a mask on the postings array indicating which elements contain all terms."""
        masks = [self.match(term) for term in tokenized_terms]
        mask = np.array([True] * len(self))
        for curr_mask in masks:
            mask = mask & curr_mask
        return mask

    def phrase_match(self, tokenized_terms, slop=1):
        """Return a boolean numpy array indicating which elements contain the given phrase."""
        # Start with docs with all terms
        mask = self.and_query(tokenized_terms)
        # For detailed documentation of this algorithm, see this ChatGPT4 discussion
        # https://chat.openai.com/share/31affaad-dc91-4757-b31c-e85bdb5a0eb6

        if np.sum(mask) == 0:
            return mask

        def vstack_with_pad(arrays, width=10):
            vstacked = -np.ones((len(arrays), width), dtype=arrays[0].dtype)
            for idx, array in enumerate(arrays):
                vstacked[idx, :len(array)] = array
            return vstacked

        # Pad for easy difference computation
        term_posns = []
        for term in tokenized_terms:
            as_array = self.positions(term, mask)
            term_posns.append(vstack_with_pad(as_array, 5))

        prior_term = term_posns[0]
        for term in term_posns[1:]:
            # Compute positional differences
            #
            # Each row of posn_diffs is a term posn diff matrix
            # Where columns are prior_term posns, rows are term posns
            # This shows every possible term diff
            #
            # Example:
            #   prior_term = array([[0, 4],[0, 4])
            #         term = array([[1, 2, 3],[1, 2, 3]])
            #
            #
            #   posn_diffs =
            #
            #     array([[ term[0] - prior_term[0], term[0] - prior_term[1] ],
            #            [ term[1] - prior_term[0], ...
            #            [ term[2] - prior_term[0], ...
            #
            #    or in our example
            #
            #     array([[ 1, -3],
            #            [ 2, -2],
            #            [ 3, -1]])
            #
            #  We care about all locations where posn == slop (or perhaps <= slop)
            #  that is term is slop away from prior_term. Usually slop == 1 (ie 1 posn away)
            #  for normal phrase matching
            #
            posn_diffs = term[:, :, np.newaxis] - prior_term[:, np.newaxis, :]

            # For > 2 terms, we need to connect a third term by making prior_term = term
            # and repeating
            #
            # BUT
            # we only want those parts of term that are adjacent to prior_term
            # before continuing, so we don't accidentally get a partial phrase
            # so we need to make sure to
            # Pad out any rows in 'term' where posn diff != slop
            # so they're not considered on subsequent iterations
            term_mask = np.any(posn_diffs == 1, axis=2)
            term[~term_mask] = -1

            # Count how many times the row term is 1 away from the col term
            per_doc_diffs = np.sum(posn_diffs == slop, axis=1)

            # Doc-wise sum to get a 'term freq' for the prior_term - term bigram
            bigram_freqs = np.sum(per_doc_diffs == slop, axis=1)

            # I _think_ last loop, bigram_freqs is the full phrase term freq

            # Update mask to eliminate any non-matches
            mask[mask] &= bigram_freqs > 0

            # Should only keep positions of 'prior term' that are adjacent to the
            # one prior to it...
            prior_term = term

        return mask
