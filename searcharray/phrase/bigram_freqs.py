"""Given two roaringish encoded arrays, count bigram matches between them."""
import numpy as np
from typing import Tuple, Optional
from searcharray.roaringish import RoaringishEncoder
import logging
from enum import Enum

from searcharray.roaringish import intersect, popcount64, merge, popcount_reduce_at, key_sum_over


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
_upper_bit = _1 << (encoder.payload_lsb_bits - _1)

MAX_POSN = encoder.max_payload


class Continuation(Enum):
    """Enum for continuation of bigram search."""
    LHS = 0
    RHS = 1
    BOTH = 2


def _any_payload_empty(enc: np.ndarray):
    """Check if the payload has any empty of enc."""
    return np.any((enc & encoder.payload_lsb_mask) == 0)


def _adj_to_phrase_freq(overlap: np.ndarray, adjacents: np.ndarray) -> np.ndarray:
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
    consecutive_ones = popcount64(consecutive_ones)
    adjacents -= -np.floor_divide(consecutive_ones, -2, dtype=np.int64)
    return adjacents


def _inner_bigram_same_term(lhs_int: np.ndarray, rhs_int: np.ndarray,
                            lhs_doc_ids: np.ndarray,
                            phrase_freqs: np.ndarray,
                            cont: Continuation = Continuation.RHS) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
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
    adjacents = popcount64(adj_count).view(np.int64)

    adjusted = _adj_to_phrase_freq(overlap, adjacents).astype(np.uint64)
    key_sum_over(lhs_doc_ids, adjusted, phrase_freqs)
    # Continue with ?? ?? foo foo
    # term_int without lsbs
    term_int_msbs = lhs_int & ~encoder.payload_lsb_mask
    rhs_cont = None
    lhs_cont = None
    # rhs_cont = term_int_msbs | encoder.payload_lsb(rhs_shift)
    rhs_cont = encoder.payload_lsb(rhs_shift & rhs_int) | term_int_msbs
    lhs_cont = term_int_msbs | encoder.payload_lsb(lhs_int & (lhs_int >> _1))

    if cont not in [Continuation.RHS, Continuation.BOTH]:
        rhs_cont = None
    if cont not in [Continuation.LHS, Continuation.BOTH]:
        lhs_cont = None

    return phrase_freqs, (lhs_cont, rhs_cont)


def _inner_bigram_freqs(lhs: np.ndarray,
                        rhs: np.ndarray,
                        lhs_int: np.ndarray,
                        rhs_int: np.ndarray,
                        phrase_freqs: np.ndarray,
                        cont: Continuation = Continuation.RHS) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Count bigram matches between two encoded arrays, within a 64 bit word with same MSBs.

    Parameters:
    -----------
    lhs: roaringish encoded posn array of left term
    rhs: roaringish encoded posn array of right term
    lhs_int: np.ndarray, intersected lhs
    rhs_int: np.ndarray, intersected rhs
    phrase_freqs: np.ndarray, preallocated phrase freqs for output
    cont_rhs: bool, whether to continue matching on the rhs or lhs

    Returns:
    --------
    count: number of matches per doc
    cont_next: the next (lhs or rhs) array to continue matching

    """
    # lhs_int, rhs_int = encoder.intersect(lhs, rhs)
    lhs_doc_ids = encoder.keys(lhs_int)
    if len(lhs_int) != len(rhs_int):
        raise ValueError("Encoding error, MSBs apparently are duplicated among your encoded posn arrays.")
    if len(lhs_int) == 0 and cont == Continuation.RHS:
        return phrase_freqs, (None, rhs_int)
    if len(lhs_int) == 0 and cont == Continuation.LHS:
        return phrase_freqs, (lhs_int, None)
    if len(lhs_int) == 0 and cont == Continuation.BOTH:
        return phrase_freqs, (lhs_int, rhs_int)

    # For perf we don't check all values, but if overlapping posns allowed in future, maybe we should
    same_term = (len(lhs_int) == len(rhs_int) and np.all(lhs_int == rhs_int))
    if same_term:
        return _inner_bigram_same_term(lhs_int, rhs_int, lhs_doc_ids, phrase_freqs, cont)

    rhs_next = None
    lhs_next = None
    overlap_bits = (lhs_int & encoder.payload_lsb_mask) & ((rhs_int & encoder.payload_lsb_mask) >> _1)

    if cont in [Continuation.RHS, Continuation.BOTH]:
        rhs_next = (overlap_bits << _1) & encoder.payload_lsb_mask
        rhs_next |= (rhs_int & (encoder.key_mask | encoder.payload_msb_mask))
    if cont in [Continuation.LHS, Continuation.BOTH]:
        lhs_next = overlap_bits.copy()
        lhs_next |= (lhs_int & (encoder.key_mask | encoder.payload_msb_mask))

    popcount_reduce_at(lhs_doc_ids, overlap_bits, phrase_freqs)
    return phrase_freqs, (lhs_next, rhs_next)


def _adjacent_bigram_freqs(lhs: np.ndarray, rhs: np.ndarray,
                           lhs_adj: np.ndarray, rhs_adj: np.ndarray,
                           phrase_freqs: np.ndarray,
                           cont: Continuation = Continuation.RHS) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Count bigram matches between two encoded arrays where they occur in adjacent 64 bit words.

    Returns:
    --------
    count: number of matches per doc
    rhs_next: the next rhs array to continue matching

    """
    lhs_doc_ids = encoder.keys(lhs_adj)
    # lhs lsb set and rhs lsb's most significant bit set
    matches = ((lhs_adj & _upper_bit) != 0) & ((rhs_adj & _1) != 0)
    unique, counts = np.unique(lhs_doc_ids[matches], return_counts=True)
    phrase_freqs[unique] += counts
    # Set lsb to 0 where no match, lsb to 1 where match
    rhs_next = None if cont == Continuation.LHS else np.asarray([], dtype=np.uint64)
    lhs_next = None if cont == Continuation.RHS else np.asarray([], dtype=np.uint64)
    if np.any(matches):
        if cont in [Continuation.RHS, Continuation.BOTH]:
            rhs_next = rhs_adj[matches]
            assert rhs_next is not None
            rhs_next = encoder.header(rhs_next) | _1
        if cont in [Continuation.LHS, Continuation.BOTH]:
            lhs_next = lhs_adj[matches]
            assert lhs_next is not None
            lhs_next = encoder.header(lhs_next) | _upper_bit
            rhs_next = None
    return phrase_freqs, (lhs_next, rhs_next)


def _set_adjbit_at_header(next_inner: np.ndarray, next_adj: np.ndarray,
                          cont: Continuation = Continuation.RHS) -> np.ndarray:
    """Merge two encoded arrays on their headers."""
    if len(next_inner) == 0:
        return next_adj
    if len(next_adj) == 0:
        return next_inner

    same_header_inner, same_header_adj = intersect(next_inner, next_adj,
                                                   mask=encoder.header_mask)
    # Set _1 on intersection
    ignore_mask = np.ones(len(next_adj), dtype=bool)
    ignore_mask[same_header_adj] = False
    if len(same_header_inner) > 0 and cont == Continuation.RHS:
        next_inner[same_header_inner] |= _1
        next_adj = next_adj[ignore_mask]
    if len(same_header_adj) > 0 and cont == Continuation.LHS:
        next_inner[same_header_inner] |= _upper_bit
        next_adj = next_adj[ignore_mask]
    return merge(next_inner, next_adj)


def bigram_freqs(lhs: np.ndarray,
                 rhs: np.ndarray,
                 phrase_freqs: np.ndarray,
                 cont: Continuation = Continuation.RHS) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Count bigram matches between two roaringish encoded posn arrays.
       Also return connection on right hand of the bigram

    Parameters:
    -----------
    lhs: roaringish encoded posn array of left term
    rhs: roaringish encoded posn array of right term
    phrase_freqs: np.ndarray, preallocated phrase freqs for output
    cont_rhs: bool, whether to continue matching on the rhs or lhs

    Returns:
    --------
    phrase_freqs: number of matches per doc (updated input)
    rhs_next: encoded array of end of each bigram
              (use as lhs on subsequent calls to continue the phrase)

    """
    # Example zooming into just one 64 bit word (this of course is repeated over
    # the entire array, for all docs and posn msbs)
    #
    # ***************************************************************************
    # "Inner" matches:
    # As roaringish arrays put posns on a 64 bit word, we first search for
    # simpler matches within 64 bit words from same docs with same posn header.
    # We do this by intersecting on the header (doc_id and posn_msb), then checking
    # the following pattern in the posn bits:
    #
    #           doc_id  posns_msb   posns_bits
    # lhs_term:   1234        101   0010001010
    # rhs_term:   1234        101   0101000100
    #                                 ^^  ^^  <- inner matches, terms next to each other
    #
    lhs_int, rhs_int, lhs_adj, rhs_adj = encoder.intersect_candidates(lhs, rhs)

    phrase_freqs, (lhs_next_inner, rhs_next_inner)\
        = _inner_bigram_freqs(lhs, rhs,
                              lhs_int, rhs_int,
                              phrase_freqs,
                              cont)
    # ***************************************************************************
    # "Adjacent" matches:
    # Of course we also need to check for matches that span two 64 bit words,
    # thats what adjacent matches do. First finding where header meets header + 1,
    # then checking the bit pattern shown below:
    #
    #           doc_id  posns_msb   posns_bits
    # lhs_term:   1234        101   1010000000
    # rhs_term:   1234        111   0001000101
    #                               ^        ^  <- adjacent matches, terms next to each other
    #                                              but note the posn_msb of lhs is
    #                                              1 less than the posn_msb of rhs
    #                                              so we detect these differently
    phrase_freqs, (lhs_next_adj, rhs_next_adj)\
        = _adjacent_bigram_freqs(lhs, rhs,
                                 lhs_adj, rhs_adj,
                                 phrase_freqs, cont)

    # ***************************************************************************
    # rhs_next is where the bigram ends on the rhs side, we can use this
    # to continue the phrase matching
    #
    # Combining the examples above:
    #     rhs_next_inner:   1234        101   0001000100
    #                                            ^    ^  <- where we had a match (inner)
    #
    # Merged with
    #     rhs_next_adj:      1234        111   0000000001
    #                                                   ^  <- where we had a match (adjacent)
    #
    # So we merge these... in this case, we have a two-value array
    #
    #     rhs_next:   1234,101,0001000100|1234,111,0000000001
    #
    #
    # Now rhs_next can be the LHS of the next call to bigram_freqs
    # to continue the phrase
    #
    rhs_next = None
    lhs_next = None
    if cont in [Continuation.RHS, Continuation.BOTH]:
        assert rhs_next_inner is not None
        assert rhs_next_adj is not None
        rhs_next = _set_adjbit_at_header(rhs_next_inner, rhs_next_adj, Continuation.RHS)
    if cont in [Continuation.LHS, Continuation.BOTH]:
        assert lhs_next_inner is not None
        assert lhs_next_adj is not None
        lhs_next = _set_adjbit_at_header(lhs_next_inner, lhs_next_adj, Continuation.LHS)

    return phrase_freqs, (lhs_next, rhs_next)
