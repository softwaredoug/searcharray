"""Encode positions in bits along with some neighboring information for wrapping.

See this notebook for motivation:

https://colab.research.google.com/drive/10tIEkdlCE_1J_CcgEcV0jkLfBc-0H4am?authuser=1#scrollTo=XWzy-n9dF3PG

"""
import numpy as np
import sortednp as snp
from copy import deepcopy
from typing import List, Optional
import numbers
import logging
from time import perf_counter


logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


DOC_ID_MASK = 0xFFFFFFF000000000
DOC_ID_BITS = 28
POSN_MSB_MASK = 0x0000000FFFFC0000
POSN_MSB_BITS = 18
POSN_LSB_MASK = 0x000000000003FFFF
POSN_LSB_BITS = 18

assert DOC_ID_BITS + POSN_MSB_BITS + POSN_LSB_BITS == 64
assert DOC_ID_MASK | POSN_MSB_MASK | POSN_LSB_MASK == 0xFFFFFFFFFFFFFFFF


def validate_posns(posns: np.ndarray):
    if not np.all(posns < 2**POSN_LSB_BITS):
        raise ValueError(f"Positions must be less than {2**POSN_LSB_BITS}")


def encode_posns(posns: np.ndarray, doc_ids: Optional[np.ndarray] = None):
    """Pack a sorted array of positions into compact bit array.

    each returned array represents a single term, with doc_id as 32 MSBs

    | 32 MSBs | 16 LSBs | 16 LSBs |
      doc_id    bits msbs posns

    for later easy intersection of 32+16 msbs, then checking for adjacent
    positions

    """
    validate_posns(posns)
    cols = posns // POSN_LSB_BITS    # Header of bit to use
    cols = cols.astype(np.uint64) << POSN_MSB_BITS
    if doc_ids is not None:
        cols |= doc_ids.astype(np.uint64) << (64 - DOC_ID_BITS)
    values = posns % POSN_LSB_BITS   # Value to encode

    change_indices = np.nonzero(np.diff(cols))[0] + 1
    change_indices = np.insert(change_indices, 0, 0)

    encoded = cols | (1 << values)
    if len(encoded) == 0:
        return encoded
    return np.bitwise_or.reduceat(encoded, change_indices)


def decode_posns(encoded):
    start = perf_counter()
    doc_ids = (encoded & DOC_ID_MASK) >> (64 - DOC_ID_BITS)
    msbs = (encoded & POSN_MSB_MASK) >> POSN_MSB_BITS
    to_concat = []
    for bit in range(POSN_LSB_BITS):
        mask = 1 << bit
        lsbs = encoded & mask
        set_lsbs = (lsbs != 0)
        this_doc_ids = doc_ids[set_lsbs]
        posns = bit + (msbs[set_lsbs] * POSN_LSB_BITS)
        doc_with_posn = np.dstack([this_doc_ids, posns])[0]
        to_concat.append(doc_with_posn)

    logger.debug(f"loop took {perf_counter() - start:.2f} seconds")
    stacked = np.vstack(to_concat)
    # Sort by doc_id, then posn
    sorted_posns = stacked[np.lexsort((stacked[:, 1], stacked[:, 0]))]
    doc_ids, idx = np.unique(sorted_posns[:, 0], return_index=True)
    grouped = np.split(sorted_posns[:, 1], idx[1:])
    as_list = list(zip(doc_ids, grouped))

    logger.debug(f"groupby took {perf_counter() - start:.2f} seconds | got {len(as_list)}")
    return as_list


def intersect_msbs(lhs: np.ndarray, rhs: np.ndarray):
    """Return the MSBs that are common to both lhs and rhs."""
    # common = np.intersect1d(lhs_msbs, rhs_msbs)
    _, (lhs_idx, rhs_idx) = snp.intersect(lhs >> POSN_LSB_BITS, rhs >> POSN_LSB_BITS, indices=True)
    # With large arrays np.isin becomes a bottleneck
    return lhs[lhs_idx], rhs[rhs_idx]


def convert_doc_ids(doc_ids):
    if isinstance(doc_ids, numbers.Number):
        return np.asarray([doc_ids], dtype=np.uint64)
    elif isinstance(doc_ids, list):
        return np.asarray(doc_ids, dtype=np.uint64)
    elif isinstance(doc_ids, np.ndarray):
        return doc_ids.astype(np.uint64)
    elif isinstance(doc_ids, range):
        return np.asarray(doc_ids, dtype=np.uint64)  # UNFORTUNATE COPY


def get_docs(encoded: np.ndarray, doc_ids: np.ndarray):
    """Get list of encoded that have positions in doc_ids."""
    doc_ids = convert_doc_ids(doc_ids)
    assert len(doc_ids.shape) == 1
    assert len(encoded.shape) == 1
    encoded_doc_ids = encoded.astype(np.uint64) >> (64 - DOC_ID_BITS)
    empty = doc_ids << (64 - DOC_ID_BITS)
    _, (idx_docs, idx_enc) = snp.intersect(doc_ids, encoded_doc_ids, indices=True,
                                           duplicates=snp.KEEP_MAX_N)

    found = encoded[idx_enc]
    empties = empty[np.isin(doc_ids, found, invert=True)]

    merged = snp.merge(found, empties, duplicates=snp.DROP)
    return merged


def inner_bigram_match(lhs, rhs):
    """Count bigram matches between two encoded arrays."""
    lhs, rhs = intersect_msbs(lhs, rhs)
    if len(lhs) != len(rhs):
        raise ValueError("Encoding error, MSBs apparently are duplicated among your encoded posn arrays.")
    rhs_next = (rhs & DOC_ID_MASK)
    rhs_next_mask = np.zeros(len(rhs), dtype=bool)
    counts = []
    # With popcount soon to be in numpy, this could potentially
    # be simply a left shift of the RHS LSB poppcount, and and a popcount
    # to count the overlaps
    for bit in range(1, POSN_LSB_BITS - 1):
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
            encoded = encode_posns(doc_ids=doc_ids, posns=posns)
            decode_again = decode_posns(encoded)
            if check:
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
        for term_id, posns in self.encoded_term_posns.items():
            encoded = self.encoded_term_posns[term_id]
            assert len(encoded.shape) == 1
            sliced_term_posns[term_id] = get_docs(encoded, doc_ids=doc_ids)

        return PosnBitArray(sliced_term_posns, doc_ids=doc_ids)

    def __getitem__(self, key):
        return self.slice(key)

    def merge(self, other):
        for term_id, posns in self.encoded_term_posns.items():
            try:
                posns_other = other.encoded_term_posns[term_id]
                self.encoded_term_posns[term_id] = snp.merge(posns, posns_other)
            except KeyError:
                pass

    def positions(self, term_id: int, key) -> List:
        # Check if key is in doc ids?
        start = perf_counter()
        doc_ids = index_range(self.doc_ids, key)
        logger.debug(f"index_range took {perf_counter() - start} seconds")
        if isinstance(doc_ids, numbers.Number):
            doc_ids = np.asarray([doc_ids])
        try:
            term_posns = get_docs(self.encoded_term_posns[term_id],
                                  doc_ids=doc_ids)
            logger.debug(f"get_docs took {perf_counter() - start} seconds")
        except KeyError:
            r_val = [np.array([], dtype=np.uint32) for doc_id in doc_ids]
            if len(r_val) == 1 and isinstance(key, numbers.Number):
                r_val = r_val[0]
            logger.debug(f"positions exit(1) took {perf_counter() - start} seconds")
            return r_val

        decoded = decode_posns(term_posns)
        logger.debug(f"decode took {perf_counter() - start} seconds")

        if len(decoded) == 0:
            return np.array([], dtype=np.uint32)
            logger.debug(f"positions exit(2) took {perf_counter() - start} seconds")
        if len(decoded) != len(doc_ids):
            # Fill non matches
            as_dict = dict(decoded)
            decs = []
            for doc_id in doc_ids:
                if doc_id in as_dict:
                    decs.append(as_dict[doc_id])
                else:
                    decs.append(np.array([], dtype=np.uint32))
            logger.debug(f"positions exit(3) took {perf_counter() - start} seconds")
            return decs
        else:
            decs = [dec[1] for dec in decoded]
            if len(decs) == 1 and isinstance(key, numbers.Number):
                decs = decs[0]
            logger.debug(f"positions exit(4) took {perf_counter() - start} seconds")
            return decs

    def insert(self, key, term_ids_to_posns):
        new_posns = PosnBitArrayBuilder()
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
