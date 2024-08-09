"""Encode positions in bits along with some neighboring information for wrapping.

See this notebook for motivation:

https://colab.research.google.com/drive/10tIEkdlCE_1J_CcgEcV0jkLfBc-0H4am?authuser=1#scrollTo=XWzy-n9dF3PG

"""
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict, Union, cast, Optional
from searcharray.roaringish import RoaringishEncoder, convert_keys, merge, intersect
from searcharray.phrase.memmap_arrays import MemoryMappedArrays, ArrayDict
from searcharray.phrase.bigram_freqs import bigram_freqs, Continuation
from searcharray.phrase.spans import span_search
import numbers
import logging
from collections import defaultdict, abc


logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


encoder = RoaringishEncoder()

# To not constantly type coerce
_64 = np.uint64(64)
_2 = np.uint64(2)
_1 = np.uint64(1)
_0 = np.uint64(0)
_neg1 = np.int64(-1)

MAX_POSN = encoder.max_payload


def trim_phrase_search(encoded_posns: List[np.ndarray]) -> List[np.ndarray]:
    """Trim long phrases by searching the rarest terms first."""

    # Start with rarest term
    shortest_keys = None
    shortest_idx = None
    min_len = 1e100
    max_len = 0
    for idx, enc_posn in enumerate(encoded_posns):
        if len(enc_posn) < min_len:
            shortest_keys = encoder.keys(enc_posn)
            shortest_idx = idx
            min_len = len(enc_posn)
        if len(enc_posn) > max_len:
            max_len = len(enc_posn)

    if shortest_keys is None:
        return encoded_posns

    for enc_posn_idx in range(len(encoded_posns)):
        if enc_posn_idx == shortest_idx:
            continue
        if len(encoded_posns[enc_posn_idx]) > (20 * min_len):
            encoded_posns[enc_posn_idx] = encoder.slice(encoded_posns[enc_posn_idx],
                                                        shortest_keys)

    return encoded_posns


def _intersect_bigram_matches(ids: Optional[np.ndarray],
                              counts: Optional[np.ndarray],
                              new_ids: np.ndarray,
                              new_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect bigram matches."""
    if ids is None or counts is None:
        return new_ids, new_counts
    else:
        # Intersect (Are these sorted?)
        is_sorted = np.all(np.diff(ids) >= 0)
        assert is_sorted
        is_sorted = np.all(np.diff(new_ids) >= 0)
        assert is_sorted

        ids_idx, new_ids_idx = intersect(ids, new_ids)
        intersected_counts = counts[ids_idx]
        counts = np.minimum(intersected_counts, new_counts[new_ids_idx])
        ids = ids[ids_idx]
        assert ids is not None
        assert counts is not None
        return ids, counts


def _compute_phrase_freqs_left_to_right(encoded_posns: List[np.ndarray],
                                        max_doc_id: np.uint64 = _0,
                                        trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phrase freqs from a set of encoded positions."""
    if len(encoded_posns) < 2:
        raise ValueError("phrase must have at least two terms")

    # Trim long phrases by searching the rarest terms first
    if trim and len(encoded_posns) > 3:
        encoded_posns = trim_phrase_search(encoded_posns)

    ids = None
    counts = None

    lhs = encoded_posns[0]
    for rhs in encoded_posns[1:]:
        # Only count the count of the last bigram (ignoring the ones where priors did not match)
        phrase_freqs, conts = bigram_freqs(lhs, rhs,
                                           cont=Continuation.RHS)
        assert conts[1] is not None
        lhs = conts[1]
        ids, counts = _intersect_bigram_matches(ids, counts, phrase_freqs[0], phrase_freqs[1])

    if ids is None or counts is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)

    return (ids, counts)


def _compute_phrase_freqs_right_to_left(encoded_posns: List[np.ndarray],
                                        max_doc_id: np.uint64 = _0,
                                        trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phrase freqs from a set of encoded positions."""
    if len(encoded_posns) < 2:
        raise ValueError("phrase must have at least two terms")

    # Trim long phrases by searching the rarest terms first
    if trim and len(encoded_posns) > 3:
        encoded_posns = trim_phrase_search(encoded_posns)

    ids = None
    counts = None

    rhs = encoded_posns[-1]
    for lhs in encoded_posns[-2::-1]:
        # Only count the count of the last bigram (ignoring the ones where priors did not match)
        phrase_freqs, conts = bigram_freqs(lhs, rhs,
                                           cont=Continuation.LHS)

        assert conts[0] is not None
        rhs = conts[0]
        ids, counts = _intersect_bigram_matches(ids, counts, phrase_freqs[0], phrase_freqs[1])

    if ids is None or counts is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)
    return (ids, counts)


def compute_phrase_freqs(encoded_posns: List[np.ndarray],
                         max_doc_id: np.uint64 = _0,
                         trim: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phrase freqs from a set of encoded positions."""
    shortest_len_index = min(enumerate(encoded_posns), key=lambda x: len(x[1]))[0]
    if shortest_len_index <= 1:
        return _compute_phrase_freqs_left_to_right(encoded_posns, trim=trim, max_doc_id=max_doc_id)
    elif shortest_len_index >= len(encoded_posns) - 2:
        return _compute_phrase_freqs_right_to_left(encoded_posns, trim=trim, max_doc_id=max_doc_id)
    else:
        # We optimize this case by going middle-out
        # We can take the min of both directions phrase freqs
        lhs_ids, lhs_counts = _compute_phrase_freqs_left_to_right(encoded_posns[:shortest_len_index], trim=trim, max_doc_id=max_doc_id)
        rhs_ids, rhs_counts = _compute_phrase_freqs_right_to_left(encoded_posns[shortest_len_index:], trim=trim, max_doc_id=max_doc_id)
        return _intersect_bigram_matches(lhs_ids, lhs_counts, rhs_ids, rhs_counts)


class PosnBitArrayFromFlatBuilder:
    """ Build from sorted array shape num terms x 3.

        0th is term id
        1st is doc id
        2nd is posn

        Sorted by term id then posns

    """

    def __init__(self, flat_array: np.ndarray, max_doc_id: int) -> None:
        self.flat_array = flat_array
        self.max_doc_id = max_doc_id

    def build(self):
        """Slice the flat array into a 2d array of doc ids and posns."""
        term_boundaries = np.argwhere(np.diff(self.flat_array[0]) > 0).flatten() + 1
        term_boundaries = np.concatenate([[_0],
                                          term_boundaries.view(np.uint64),
                                          np.asarray([len(self.flat_array[1])], dtype=np.uint64)])
        term_boundaries = term_boundaries.view(np.uint64)

        encoded, enc_term_boundaries = encoder.encode(keys=self.flat_array[1].view(np.uint64),
                                                      boundaries=term_boundaries[:-1],
                                                      payload=self.flat_array[2].view(np.uint64))
        if len(encoded) == 0:
            return PosnBitArray(ArrayDict(), self.max_doc_id)
        term_ids = self.flat_array[0][term_boundaries[:-1]]

        encoded_term_posns = ArrayDict.from_array_with_boundaries(encoded,
                                                                  boundaries=enc_term_boundaries,
                                                                  ids=term_ids)
        largest_doc_id_with_term = np.max(self.flat_array[1])
        assert self.max_doc_id >= largest_doc_id_with_term
        return PosnBitArray(encoded_term_posns, self.max_doc_id)


class PosnBitArrayBuilder:

    def __init__(self):
        self.term_posns = defaultdict(list)
        self.term_posn_doc_ids = defaultdict(list)
        self.max_doc_id = 0

    def add_posns(self, doc_id: int, term_id: int, posns: List[int]):
        doc_ids = [doc_id] * len(posns)
        self.term_posns[term_id].extend(posns)
        self.term_posn_doc_ids[term_id].extend(doc_ids)

    def ensure_capacity(self, doc_id):
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def build(self, check=False):
        encoded_term_posns = ArrayDict()
        for term_id, posns in self.term_posns.items():
            if len(posns) == 0:
                posns = np.asarray([], dtype=np.uint32).flatten()
            elif isinstance(posns, list):
                posns_arr = np.asarray(posns, dtype=np.uint32).flatten()
                posns = posns_arr
            doc_ids = self.term_posn_doc_ids[term_id]
            if isinstance(doc_ids, list):
                doc_ids = np.asarray(doc_ids, dtype=np.uint32)
            encoded = encoder.encode(keys=doc_ids, payload=posns)
            if check:
                decode_again = encoder.decode(encoded)
                docs_to_posns = dict(decode_again)
                doc_ids_again = []
                posns_again = []
                for doc_id, posns_dec in docs_to_posns.items():
                    for posn in posns_dec:
                        doc_ids_again.append(doc_id)
                        posns_again.append(posn)
                assert np.array_equal(doc_ids_again, doc_ids)
                assert np.array_equal(posns, posns_again)
            encoded_term_posns[term_id] = encoded

        return PosnBitArray(encoded_term_posns, self.max_doc_id)


class PosnBitArrayAlreadyEncBuilder:

    def __init__(self):
        self.encoded_term_posns = ArrayDict()
        self.max_doc_id = 0

    def add_posns(self, doc_id: int, term_id: int, posns):
        self.encoded_term_posns[term_id] = posns

    def ensure_capacity(self, doc_id):
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def build(self, check=False):
        return PosnBitArray(self.encoded_term_posns, self.max_doc_id)


def index_range(rng, key):
    if key is None:
        return rng
    if isinstance(rng, np.ndarray):
        return rng[key]

    if isinstance(key, slice):
        return rng[key]
    elif isinstance(key, numbers.Number):
        return rng[key]
    elif isinstance(key, np.ndarray):
        try:
            # UNFORTUNATE COPY
            r_val = np.asarray(list(rng))[key]
            return r_val
        except IndexError as e:
            raise e
    # Last resort
    # UNFORTUNATE COPY
    # Here probably elipses or a tuple of various things
    return np.asarray(list(rng))[key]


class FilteredPosns(abc.Mapping):
    """When someone slices, this lets us avoid repeating constantly
    slicing the keys out of the encoded positions."""

    def __init__(self, base, doc_ids):
        self.base = base
        self.sliced = {}
        self.doc_ids = doc_ids

    def __getitem__(self, key):
        """Slice with encoder, cache, and return."""
        if key in self.sliced:
            return self.sliced[key]

        sliced = encoder.slice(self.base[key],
                               keys=self.doc_ids)
        self.sliced[key] = sliced
        return sliced

    def __setitem__(self, key, value):
        self.base[key] = value

    def __iter__(self):
        return iter(self.doc_ids)

    def __len__(self):
        return len(self.doc_ids)


class PosnBitArray:

    def __init__(self, encoded_term_posns: Union[ArrayDict, FilteredPosns], max_doc_id: int,
                 cache_gt_than=25):
        self.encoded_term_posns = encoded_term_posns
        self.max_doc_id = max_doc_id
        self.docfreq_cache : Dict[int, np.uint64] = {}
        self.termfreq_cache : Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.cache_gt_than = cache_gt_than

    def __repr__(self):
        return f"""PosnBitArray(encoded_term_posns={self.encoded_term_posns}, max_doc_id={self.max_doc_id})"""

    def memmap(self, data_dir):
        if self.encoded_term_posns:
            self.encoded_term_posns = MemoryMappedArrays(data_dir, self.encoded_term_posns)

    def warm(self):
        """Warm tf / df cache of most common terms."""
        for term_id, encoded in self.encoded_term_posns.items():
            if len(encoded) > 255:
                self.docfreq(term_id)
                self.termfreqs(term_id)

    def filter(self, doc_ids) -> "PosnBitArray":
        """Filter my doc ids to only those in doc_ids."""
        enc_term_posns = self.encoded_term_posns
        if isinstance(enc_term_posns, FilteredPosns):
            self.encoded_term_posns = enc_term_posns.base
        filtered = FilteredPosns(self.encoded_term_posns, doc_ids)
        new_bit_array = PosnBitArray(filtered, self.max_doc_id)
        return new_bit_array

    def _reset_filter(self):
        if isinstance(self.encoded_term_posns, FilteredPosns):
            self.encoded_term_posns = self.encoded_term_posns.base

    def clear_cache(self):
        self.docfreq_cache = {}
        self.termfreq_cache = {}
        self._reset_filter()

    def copy(self):
        new = PosnBitArray(deepcopy(self.encoded_term_posns), self.max_doc_id)
        return new

    def concat(self, other):
        """Merge other into self.

        Assumes other's doc ids are not overlapping with self's doc ids.
        """
        if self.encoded_term_posns == {}:
            self.encoded_term_posns = other.encoded_term_posns
            self.max_doc_id = other.max_doc_id
            self.clear_cache()
            return self
        self.encoded_term_posns = ArrayDict.concat(self.encoded_term_posns, other.encoded_term_posns)
        self.max_doc_id = max(self.max_doc_id, other.max_doc_id)
        self.clear_cache()

    def slice(self, key):
        sliced_term_posns = {}
        doc_ids = convert_keys(key)
        max_doc_id = np.max(doc_ids)
        for term_id, posns in self.encoded_term_posns.items():
            encoded = self.encoded_term_posns[term_id]
            assert len(encoded.shape) == 1
            sliced_term_posns[term_id] = encoder.slice(encoded, keys=doc_ids)

        return PosnBitArray(sliced_term_posns, max_doc_id)

    def __getitem__(self, key):
        return self.slice(key)

    def merge(self, other):
        # Unique terms
        unique_terms = set(self.encoded_term_posns.keys()).union(set(other.encoded_term_posns.keys()))

        for term_id in unique_terms:
            if term_id not in other.encoded_term_posns:
                continue
            elif term_id not in self.encoded_term_posns:
                self.encoded_term_posns[term_id] = other.encoded_term_posns[term_id]
            else:
                posns_self = self.encoded_term_posns[term_id]
                posns_other = other.encoded_term_posns[term_id]
                self.encoded_term_posns[term_id] = merge(posns_self, posns_other)
        self.max_doc_id = self.max_doc_id + other.max_doc_id
        self.clear_cache()

    def doc_encoded_posns(self, term_id: int, doc_id: int) -> np.ndarray:
        term_posns = encoder.slice(self.encoded_term_posns[term_id],
                                   keys=np.asarray([doc_id], dtype=np.uint64))
        return term_posns

    def empty_buffer(self):
        return np.zeros(int(self.max_doc_id + 1), dtype=np.float32)

    def phrase_freqs(self, term_ids: List[int],
                     slop: int = 0,
                     doc_ids: Optional[np.ndarray] = None,
                     min_posn: Optional[int] = None,
                     max_posn: Optional[int] = None) -> np.ndarray:
        phrase_freqs = self.empty_buffer()

        if len(term_ids) < 2:
            raise ValueError("Must have at least two terms")
        if phrase_freqs.shape[0] == self.max_doc_id + 1 and min_posn is None and max_posn is None and doc_ids is None:
            enc_term_posns = [self.encoded_term_posns[term_id] for term_id in term_ids]
        else:
            keys = None
            if doc_ids is not None:
                keys = doc_ids.view(np.uint64)

            enc_term_posns = [encoder.slice(self.encoded_term_posns[term_id],
                                            keys=keys,
                                            min_payload=min_posn,
                                            max_payload=max_posn) for term_id in term_ids]

        if slop == 0:
            ids, counts = compute_phrase_freqs(enc_term_posns, max_doc_id=np.uint64(self.max_doc_id))
            phrase_freqs[ids] = counts
            return phrase_freqs
        else:
            ids, counts = span_search(enc_term_posns, slop)
            phrase_freqs[ids] = counts
            return phrase_freqs

    def positions(self, term_id: int, doc_ids) -> Union[List[np.ndarray], np.ndarray]:
        if isinstance(doc_ids, numbers.Number):
            doc_ids = np.asarray([doc_ids])

        try:
            np_doc_ids = convert_keys(doc_ids)
            term_posns = encoder.slice(self.encoded_term_posns[term_id],
                                       keys=np_doc_ids)
        except KeyError:
            r_val = [np.array([], dtype=np.uint32) for doc_id in doc_ids]
            if len(r_val) == 1 and len(doc_ids) == 1 and isinstance(doc_ids[0], numbers.Number):
                return [r_val[0]]
            return r_val

        decoded = encoder.decode(encoded=term_posns, get_keys=True)

        if len(decoded) == 0:
            return [np.array([], dtype=np.uint32)]
        if len(decoded) != len(doc_ids):
            # Fill non matches
            decoded = cast(List[Tuple[np.uint64, np.ndarray]], decoded)
            as_dict: Dict[np.uint64, np.ndarray] = dict(decoded)
            decs = []
            for doc_id in doc_ids:
                if doc_id in as_dict:
                    decs.append(as_dict[doc_id])
                else:
                    decs.append(np.array([], dtype=np.uint32))
            return decs
        else:
            decs = [dec[1] for dec in decoded]
            return decs

    def termfreqs(self, term_id: int,
                  doc_ids: Optional[np.ndarray] = None,
                  min_posn: Optional[int] = None,
                  max_posn: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Count term freqs using unique positions."""
        if doc_ids is None and max_posn is None and min_posn is None:
            return self._termfreqs_with_cache(term_id)

        encoded = self.encoded_term_posns[term_id]
        term_posns = encoded
        term_posns = encoder.slice(encoded,
                                   keys=doc_ids,
                                   min_payload=min_posn,
                                   max_payload=max_posn)

        return self._computed_term_freqs(term_posns)

    def _computed_term_freqs(self, term_posns) -> Tuple[np.ndarray, np.ndarray]:
        return encoder.num_values_per_key(term_posns)

    def _termfreqs_with_cache(self, term_id: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self.termfreq_cache[term_id]
        except KeyError:
            term_posns = self.encoded_term_posns[term_id]
            doc_ids, term_freqs = self._computed_term_freqs(term_posns)
            if self._is_cached(term_id):
                self.termfreq_cache[term_id] = (doc_ids, term_freqs)
            return doc_ids, term_freqs

    def _is_cached(self, term_id: int) -> bool:
        return term_id in self.docfreq_cache

    def _docfreq_from_cache(self, term_id: int) -> np.uint64:
        return self.docfreq_cache[term_id]

    def _maybe_cache_docfreq(self, term_id: int, docfreq: np.uint64):
        if len(self.encoded_term_posns[term_id]) > self.cache_gt_than:
            self.docfreq_cache[term_id] = docfreq

    def docfreq(self, term_id: int) -> np.uint64:
        try:
            return self.docfreq_cache[term_id]
        except KeyError:
            encoded = self.encoded_term_posns[term_id]
            docfreq = np.uint64(encoder.keys_unique(encoded).size)
            self._maybe_cache_docfreq(term_id, docfreq)
            return docfreq

    def insert(self, key, term_ids_to_posns, is_encoded=False):
        new_posns = PosnBitArrayBuilder()
        if is_encoded:
            new_posns = PosnBitArrayAlreadyEncBuilder()
        max_doc_id = 0
        for doc_id, new_posns_row in enumerate(term_ids_to_posns):
            for term_id, positions in new_posns_row:
                new_posns.add_posns(doc_id, term_id, positions)
            max_doc_id = max(doc_id, max_doc_id)
            new_posns.max_doc_id = max_doc_id
        ins_arr = new_posns.build()
        self.merge(ins_arr)

    @property
    def nbytes(self):
        arr_bytes = 0
        for doc_id, posn in self.encoded_term_posns.items():
            arr_bytes += posn.nbytes
        for term_id, (doc_ids, term_freqs) in self.termfreq_cache.items():
            arr_bytes += doc_ids.nbytes
            arr_bytes += term_freqs.nbytes
        for term_id, docfreq in self.docfreq_cache.items():
            arr_bytes += docfreq.nbytes
        return arr_bytes
