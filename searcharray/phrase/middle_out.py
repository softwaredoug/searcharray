"""Encode positions in bits along with some neighboring information for wrapping.

See this notebook for motivation:

https://colab.research.google.com/drive/10tIEkdlCE_1J_CcgEcV0jkLfBc-0H4am?authuser=1#scrollTo=XWzy-n9dF3PG

"""
import numpy as np
import sortednp as snp
from copy import deepcopy
from typing import List
from searcharray.utils.roaringish import RoaringishEncoder, convert_keys
import numbers
import logging

from searcharray.utils.bitcount import bit_count64


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


def inner_bigram_match(lhs, rhs):
    """Count bigram matches between two encoded arrays."""
    lhs, rhs = encoder.intersect(lhs, rhs)
    if len(lhs) != len(rhs):
        raise ValueError("Encoding error, MSBs apparently are duplicated among your encoded posn arrays.")
    rhs_next = (rhs & encoder.key_mask)
    rhs_next_mask = np.zeros(len(rhs), dtype=bool)
    counts = []
    # With popcount soon to be in numpy, this could potentially
    # be simply a left shift of the RHS LSB poppcount, and and a popcount
    # to count the overlaps
    for bit in range(1, encoder.payload_lsb_bits - 1):
        lhs_mask = 1 << bit
        rhs_mask = 1 << (bit + 1)
        lhs_set = (lhs & lhs_mask) != 0
        rhs_set = (rhs & rhs_mask) != 0

        matches = lhs_set & rhs_set
        rhs_next[matches] |= rhs_mask
        rhs_next_mask |= matches
        counts.append(np.count_nonzero(matches))
    return np.sum(counts), rhs_next[rhs_next_mask]


class PosnBitArrayBuilder:

    def __init__(self):
        self.term_posns = {}
        self.term_posn_doc_ids = {}
        self.max_doc_id = 0

    def add_posns(self, doc_id: int, term_id: int, posns):
        if len(posns.shape) != 1:
            raise ValueError("posns must be a 1D array")
        if term_id not in self.term_posns:
            self.term_posns[term_id] = []
            self.term_posn_doc_ids[term_id] = []
        doc_ids = [doc_id] * len(posns)
        self.term_posns[term_id].extend(posns.tolist())
        self.term_posn_doc_ids[term_id].extend(doc_ids)
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def ensure_capacity(self, doc_id):
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def build(self, check=False):
        encoded_term_posns = {}
        for term_id, posns in self.term_posns.items():
            if len(posns) == 0:
                posns = np.asarray([], dtype=np.uint32).flatten()
            elif isinstance(posns, list):
                posns_arr = np.asarray(posns, dtype=np.uint32)
                assert len(posns_arr.shape) == 1
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

        return PosnBitArray(encoded_term_posns, range(0, self.max_doc_id + 1))


class PosnBitArrayAlreadyEncBuilder:

    def __init__(self):
        self.encoded_term_posns = {}
        self.max_doc_id = 0

    def add_posns(self, doc_id: int, term_id: int, posns):
        self.encoded_term_posns[term_id] = posns

    def ensure_capacity(self, doc_id):
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def build(self, check=False):
        return PosnBitArray(self.encoded_term_posns, range(0, self.max_doc_id + 1))


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


class PosnBitArray:

    def __init__(self, encoded_term_posns, doc_ids):
        self.encoded_term_posns = encoded_term_posns
        self.doc_ids = doc_ids

    def copy(self):
        new = PosnBitArray(deepcopy(self.encoded_term_posns),
                           self.doc_ids)
        return new

    def slice(self, key):
        sliced_term_posns = {}
        doc_ids = index_range(self.doc_ids, key)
        np_doc_ids = convert_keys(doc_ids)
        for term_id, posns in self.encoded_term_posns.items():
            encoded = self.encoded_term_posns[term_id]
            assert len(encoded.shape) == 1
            sliced_term_posns[term_id] = encoder.slice(encoded, keys=np_doc_ids)

        return PosnBitArray(sliced_term_posns, doc_ids=doc_ids)

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
                self.encoded_term_posns[term_id] = snp.merge(posns_self, posns_other)

    def doc_encoded_posns(self, term_id: int, doc_id: int) -> List:
        term_posns = encoder.slice(self.encoded_term_posns[term_id],
                                   keys=np.asarray([doc_id], dtype=np.uint64))
        return term_posns

    def positions(self, term_id: int, key) -> List:
        # Check if key is in doc ids?
        doc_ids = index_range(self.doc_ids, key)
        if isinstance(doc_ids, numbers.Number):
            doc_ids = np.asarray([doc_ids])

        try:
            np_doc_ids = convert_keys(doc_ids)
            term_posns = encoder.slice(self.encoded_term_posns[term_id],
                                       keys=np_doc_ids)
        except KeyError:
            r_val = [np.array([], dtype=np.uint32) for doc_id in doc_ids]
            if len(r_val) == 1 and isinstance(key, numbers.Number):
                r_val = r_val[0]
            return r_val

        decoded = encoder.decode(encoded=term_posns, get_keys=True)

        if len(decoded) == 0:
            return np.array([], dtype=np.uint32)
        if len(decoded) != len(doc_ids):
            # Fill non matches
            as_dict = dict(decoded)
            decs = []
            for doc_id in doc_ids:
                if doc_id in as_dict:
                    decs.append(as_dict[doc_id])
                else:
                    decs.append(np.array([], dtype=np.uint32))
            return decs
        else:
            decs = [dec[1] for dec in decoded]
            if len(decs) == 1 and isinstance(key, numbers.Number):
                decs = decs[0]
            return decs

    def termfreqs(self, term_id: int, doc_ids: np.ndarray) -> np.ndarray:
        """Count term freqs using unique positions."""
        encoded = self.encoded_term_posns[term_id]
        term_posns = encoder.slice(encoded,
                                   keys=doc_ids.astype(np.uint64))
        doc_ids = encoder.keys(term_posns)
        change_indices = np.nonzero(np.diff(doc_ids))[0]
        change_indices = np.concatenate(([0], change_indices + 1))
        posns = encoded & encoder.payload_lsb_mask
        bit_counts = bit_count64(posns)

        term_freqs = np.add.reduceat(bit_counts, change_indices)
        return doc_ids, term_freqs

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
        return arr_bytes

    def phrase_freqs(self, terms: List[int]):
        """Return the phrase frequencies for a list of terms."""
        raise NotImplementedError("Not yet implemented")
