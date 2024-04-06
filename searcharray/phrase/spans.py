"""Find span matches up to PAYLOAD_LSB bits span distance."""
import numpy as np
from typing import List
from searcharray.roaringish import intersect, adjacent
from searcharary.roaringish import intersect_all
from searcharary.roaringish.roaringish import RoaringishEncoder

encoder = RoaringishEncoder()

# To not constantly type coerce
_64 = np.uint64(64)
_2 = np.uint64(2)
_1 = np.uint64(1)
_0 = np.uint64(0)
_neg1 = np.int64(-1)
_upper_bit = _1 << (encoder.payload_lsb_bits - _1)

# ---
# How to find max spans in roaringish
#                   MSB
#     term A     |10|       | 12|  13|           | 18    * Intersect A+B -> 10,12
#     term B     |10|       | 12|    | 14  15 17 |     * RHS intersect -> 13 -> 14
#     term C        | 11               14
#     term D     10 |                      15
#     term E        | 11      12
#     ------
#
#
#     term A     |10| 11 | 12|  13|           * Intersect A+B -> 10,12
#     term B     |10| 11 | 12|  13|           * Intersect A+B -> 10,12
#     term C   9 |  | 11 |
#
#  * every intersect LHS and RHS are possible
#  * every adjacent only in between is possible?
def _intersect_all(posns_encoded: List[np.ndarray]) -> np.ndarray:
    """Intersect all encoded positions at roaringish MSBs."""
    if len(posns_encoded) < 2:
        raise ValueError("Need at least two positions to intersect")

    lhs_int = posns_encoded[0]
    lhs_to_left = posns_encoded[0]
    lhs_to_right = posns_encoded[0]

    for term_idx, posns_next in enumerate(posns_encoded[1:]):
        _, rhs_int = intersect(lhs_int, posns_next, mask=encoder.header_mask)
        _, rhs_ls = adjacent(lhs_to_left, posns_next, mask=encoder.header_mask)
        rhs_rs, _ = adjacent(lhs_to_right, posns_next, mask=encoder.header_mask)

        # Update LHS to rhs_int + rhs_ls + rhs_rs indices
        lhs_int = posns_next[rhs_int]
        lhs_to_left = posns_next[rhs_ls]
        lhs_to_right = posns_next[rhs_rs]

    # Get active MSBs now
    # Intersect all the rest to grab them


    return np.merge([lhs_int, lhs_to_left, lhs_to_right])


def inner_span_candidates(posns_encoded: List[np.ndarray], slop) -> np.ndarray:
    """Find inner span candidates given slop and encoded positions."""
    all_intersected = _intersect_all(posns_encoded)

    # Or each together
    orred = np.zeros_like(all_intersected, dtype=np.uint64)
    for intersected in all_intersected:
        orred |= intersected

    # Find gaps of 0s >= slop to split into candidate spans
