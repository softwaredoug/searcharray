"""Given two roaringish encoded arrays, count bigram matches between them."""
import numpy as np
from typing import Tuple
from searcharray.utils.roaringish import RoaringishEncoder
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

# To not constantly type coerce
_64 = np.uint64(64)
_2 = np.uint64(2)
_1 = np.uint64(1)
_0 = np.uint64(0)
_neg1 = np.int64(-1)

MAX_POSN = encoder.max_payload


def adj_to_phrase_freq(overlap: np.ndarray, adjacents: np.ndarray) -> np.ndarray:
    """Adjust phrase freqs for adjacent matches."""
    # ?? 1 1 1 0 1
    # we need to treat the 2nd consecutive 1 as 'not a match'
    # and also update 'term' to not include it
    # Search for foo foo
    # foo foo -> phrase freq 1
    # foo foo foo -> 2 adjs, phrase freqs = 1
    # foo foo foo foo -> 3 adjs, phrase freqs = 2
    # [foo foo] [foo foo] foo -> 4 adjs, phrase freqs = 2
    # [foo foo] [foo foo] mal [foo foo] -> 4 adjs, phrase freqs = 3
    consecutive_ones = encoder.payload_lsb(overlap & (overlap << _1))
    consecutive_ones = bit_count64(consecutive_ones)
    adjacents -= -np.floor_divide(consecutive_ones, -2).astype(np.int64)
    return adjacents


def inner_bigram_same_term(lhs_int: np.ndarray, rhs_int: np.ndarray,
                           lhs_doc_ids: np.ndarray,
                           phrase_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count bigram matches when lhs / rhs are same term.

    Despite being the same term, its possible lhs_int != rhs_int due to lhs corresponding
    to a past match of identical term."""
    # Shift RHS to left by 1
    # 3  2   1   0      3    2   1   0
    # ?? foo foo foo -> foo foo foo ??
    rhs_shift = rhs_int << _1
    # Now we have unshifted and shifted to count adjacents:
    # foo foo foo ?? & ?? foo foo foo
    # ->  ?? foo foo ?? or two times term is adjacent
    # Count these bits...
    overlap = lhs_int & rhs_shift
    adj_count = encoder.payload_lsb(overlap)
    adjacents = bit_count64(adj_count).astype(np.int64)
    phrase_freqs[lhs_doc_ids] += adj_to_phrase_freq(overlap, adjacents)
    # Continue with ?? ?? foo foo
    # term_int without lsbs
    term_int_msbs = lhs_int & ~encoder.payload_lsb_mask
    term_cont = term_int_msbs | encoder.payload_lsb(rhs_shift)
    return phrase_freqs, term_cont


def inner_bigram_freqs(lhs: np.ndarray, rhs: np.ndarray,
                       phrase_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count bigram matches between two encoded arrays, within a 64 bit word with same MSBs.

    Returns:
    --------
    count: number of matches per doc
    rhs_next: the next rhs array to continue matching

    """
    lhs_int, rhs_int = encoder.intersect(lhs, rhs)
    lhs_doc_ids = encoder.keys(lhs_int)
    if len(lhs_int) != len(rhs_int):
        raise ValueError("Encoding error, MSBs apparently are duplicated among your encoded posn arrays.")
    if len(lhs_int) == 0:
        return phrase_freqs, rhs_int
    # For perf we don't check all values, but if overlapping posns allowed in future, maybe we should
    same_term = (len(lhs_int) == len(rhs_int) and lhs_int[0] == rhs_int[0])
    if same_term:
        return inner_bigram_same_term(lhs_int, rhs_int, lhs_doc_ids, phrase_freqs)

    overlap_bits = (lhs_int & encoder.payload_lsb_mask) & ((rhs_int & encoder.payload_lsb_mask) >> _1)
    rhs_next2 = (overlap_bits << _1) & encoder.payload_lsb_mask
    rhs_next2 |= (rhs_int & (encoder.key_mask | encoder.payload_msb_mask))
    phrase_freqs2 = phrase_freqs.copy()
    matches2 = overlap_bits > 0
    if np.any(matches2):
        transitions = np.argwhere(np.diff(lhs_doc_ids[matches2]) != 0).flatten() + 1
        transitions = np.insert(transitions, 0, 0)
        counted_bits = bit_count64(overlap_bits[matches2])
        reduced = np.add.reduceat(counted_bits,
                                  transitions)
        phrase_freqs2[np.unique(lhs_doc_ids[matches2])] += reduced
    return phrase_freqs2, rhs_next2


def adjacent_bigram_freqs(lhs: np.ndarray, rhs: np.ndarray,
                          phrase_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count bigram matches between two encoded arrays where they occur in adjacent 64 bit words.

    Returns:
    --------
    count: number of matches per doc
    rhs_next: the next rhs array to continue matching

    """
    lhs_int, rhs_int = encoder.intersect_rshift(lhs, rhs, rshift=_neg1)
    lhs_doc_ids = encoder.keys(lhs_int)
    # lhs lsb set and rhs lsb's most significant bit set
    upper_bit = _1 << (encoder.payload_lsb_bits - _1)
    matches = ((lhs_int & upper_bit) != 0) & ((rhs_int & _1) != 0)
    unique, counts = np.unique(lhs_doc_ids[matches], return_counts=True)
    phrase_freqs[unique] += counts
    rhs_next = rhs_int
    rhs_next[~matches] |= ~encoder.payload_lsb_mask
    rhs_next[matches] |= (encoder.payload_lsb_mask & _1)
    return phrase_freqs, rhs_next


def bigram_freqs(lhs: np.ndarray, rhs: np.ndarray, phrase_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count bigram matches between two encoded arrays.

    Returns:
    --------
    count: number of matches per doc
    rhs_next: encoded array of end of each bigram
              (use as lhs on subsequent calls to continue the phrase)

    """
    # Combine lhs and rhs matches from two strategies
    phrase_freqs, rhs_next_inner = inner_bigram_freqs(lhs, rhs, phrase_freqs)
    # print("--- pfi", phrase_freqs)
    # print("--- coi", encoder.decode(rhs_next_inner))
    phrase_freqs, rhs_next_adj = adjacent_bigram_freqs(lhs, rhs, phrase_freqs)
    # print("--- pfa", phrase_freqs)
    # print("--- coa", encoder.decode(rhs_next_adj))
    rhs_next = np.sort(np.concatenate([rhs_next_inner, rhs_next_adj]))

    return phrase_freqs, rhs_next
