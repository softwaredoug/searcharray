"""Encode positions in bits along with some neighboring information for wrapping."""
import numpy as np
import sortednp as snp
from copy import deepcopy
import numbers


def validate_posns(posns: np.ndarray):
    if not np.all(posns < 65536):
        raise ValueError("Positions must be less than 65536")


def encode_posns(doc_ids: np.ndarray, posns: np.ndarray):
    """Pack a sorted array of positions into compact bit array.

    each returned array represents a single term, with doc_id as 32 MSBs

    | 32 MSBs | 16 LSBs | 16 LSBs |
      doc_id    bits msbs posns

    for later easy intersection of 32+16 msbs, then checking for adjacent
    positions

    """
    validate_posns(posns)
    cols = posns // 16    # Header of bit to use
    cols = cols.astype(np.uint64) << 16
    cols |= doc_ids.astype(np.uint64) << 32
    values = posns % 16   # Value to encode

    change_indices = np.nonzero(np.diff(cols))[0] + 1
    change_indices = np.insert(change_indices, 0, 0)

    encoded = cols | (1 << values)
    if len(encoded) == 0:
        return encoded
    return np.bitwise_or.reduceat(encoded, change_indices)


def decode_posns(encoded: np.ndarray):
    """Unpack bit packed positions into docids, position tuples."""
    # Get 16 MSBs
    doc_ids = (encoded & 0xFFFFFFFF00000000) >> 32

    docs_diff = np.diff(doc_ids)
    split_at = np.argwhere(docs_diff > 0).flatten() + 1

    # encoded &= 0x00000000FFFFFFFF
    enc_per_doc = np.split(encoded, split_at)

    doc_posns = []
    for idx, enc_in_doc in enumerate(enc_per_doc):
        # Mask each lsb compute its actual position
        # by adding the msb
        posn_arrays = []
        if len(enc_in_doc) == 0:
            continue

        doc_id = ((enc_in_doc & 0xFFFFFFFF00000000) >> 32)[0]
        msbs = (enc_in_doc & 0x00000000FFFF0000) >> 16
        for bit in range(16):
            mask = 1 << bit
            lsbs = enc_in_doc & mask
            set_lsbs = lsbs != 0
            posn_arrays.append(bit + (msbs[set_lsbs] * 16))
        all_posns = np.concatenate(posn_arrays)
        doc_posns.append((doc_id, all_posns))
    return doc_posns


def intersect_msbs(lhs: np.ndarray, rhs: np.ndarray):
    """Return the MSBs that are common to both lhs and rhs."""
    lhs_msbs = lhs >> 16
    rhs_msbs = rhs >> 16
    # Possible speedup https://gitlab.sauerburger.com/frank/sortednp
    # common = np.intersect1d(lhs_msbs, rhs_msbs)
    common = snp.intersect(lhs_msbs, rhs_msbs)
    return lhs[np.isin(lhs_msbs, common)], rhs[np.isin(rhs_msbs, common)]


def convert_doc_ids(doc_ids):
    if isinstance(doc_ids, numbers.Number):
        return np.asarray([doc_ids], dtype=np.uint64)
    elif isinstance(doc_ids, list):
        return np.asarray(doc_ids, dtype=np.uint64)
    elif isinstance(doc_ids, np.ndarray):
        return doc_ids.astype(np.uint64)
    elif isinstance(doc_ids, range):
        return np.asarray(doc_ids, dtype=np.uint64)  # UNFORTUNATE COPY
    else:
        import pdb; pdb.set_trace()


def get_docs(encoded: np.ndarray, doc_ids: np.ndarray):
    """Get list of encoded that have positions in doc_ids."""
    doc_ids = convert_doc_ids(doc_ids)
    encoded_doc_ids = encoded.astype(np.uint64) >> 32
    try:
        empty = doc_ids << 32
    except TypeError:
        import pdb; pdb.set_trace()
    # common = np.intersect1d(doc_ids, encoded_doc_ids)
    common = snp.intersect(doc_ids, encoded_doc_ids)

    idx_enc = np.isin(encoded_doc_ids, common)
    idx_docs = np.isin(doc_ids, common)
    found = encoded_doc_ids[idx_enc]
    empties = empty[~idx_docs]

    merged = snp.merge(found, empties)
    return merged


def inner_bigram_match(encoded1: np.ndarray, encoded2: np.ndarray):
    """Count bigram matches between two encoded arrays."""
    lhs, rhs = intersect_msbs(encoded1, encoded2)
    counts = []
    for bit in range(1, 15):
        lhs_mask = 1 << (bit + 1)
        rhs_mask = 1 << bit

        lhs_set = (lhs & lhs_mask) != 0
        rhs_set = (rhs & rhs_mask) != 0

        matches = lhs_set & rhs_set
        counts.append(np.count_nonzero(matches))
    return np.sum(counts)


class PosnBitArrayBuilder:

    def __init__(self):
        self.term_posns = {}
        self.term_posn_doc_ids = {}
        self.max_doc_id = 0

    def add_posns(self, doc_id: int, term_id: int, posns):
        if term_id not in self.term_posns:
            self.term_posns[term_id] = []
            self.term_posn_doc_ids[term_id] = []
        doc_ids = [doc_id] * len(posns)
        self.term_posns[term_id].extend(posns)
        self.term_posn_doc_ids[term_id].extend(doc_ids)
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def ensure_capacity(self, doc_id):
        self.max_doc_id = max(self.max_doc_id, doc_id)

    def build(self):
        encoded_term_posns = {}
        for term_id, posns in self.term_posns.items():
            if isinstance(posns, list):
                posns = np.asarray(posns, dtype=np.uint32)
            doc_ids = self.term_posn_doc_ids[term_id]
            if isinstance(doc_ids, list):
                doc_ids = np.asarray(doc_ids, dtype=np.uint32)
            encoded = encode_posns(doc_ids, posns)
            encoded_term_posns[term_id] = encoded

        return PosnBitArray(encoded_term_posns, range(0, self.max_doc_id + 1))


def index_range(rng, key):
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
        # import pdb; pdb.set_trace()
        doc_ids = index_range(self.doc_ids, key)
        for term_id, posns in self.encoded_term_posns.items():
            encoded = self.encoded_term_posns[term_id]
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

    def positions(self, term_id, key):
        # Check if key is in doc ids?
        doc_ids = index_range(self.doc_ids, key)
        if isinstance(doc_ids, numbers.Number):
            doc_ids = [doc_ids]
        try:
            term_posns = get_docs(self.encoded_term_posns[term_id],
                                  doc_ids=np.asarray(doc_ids))
        except KeyError:
            return [np.array([], dtype=np.uint32) for doc_id in doc_ids]
        decoded = decode_posns(term_posns)

        decs = [dec[1] for dec in decoded]
        return decs

    def insert(self, key, term_ids_to_posns):
        new_posns = PosnBitArrayBuilder()
        # PROBABLY not doc id
        max_doc_id = 0
        for doc_id, new_posns_row in enumerate(term_ids_to_posns):
            for term_id, positions in new_posns_row:
                # Need to get a doc id from each key
                # Can I do this without storing the doc ids?
                # update_docs = self.posns_mats[mat_idx][key]
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
