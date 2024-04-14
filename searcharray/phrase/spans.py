"""Find span matches up to PAYLOAD_LSB bits span distance."""
#  Colab notebooks
#    https://colab.research.google.com/drive/1H4g7eeaYaHT1dJ5dmcmZUcwKNT-14tYs?authuser=1
#  Chat GPT convo
#   https://chat.openai.com/share/b97c61e6-95d0-40e7-a8cc-447dc1495314
import numpy as np
from typing import List
from searcharray.roaringish import intersect, adjacent, merge, span_search as r_span_search
# from searcharary.roaringish import intersect_all
from searcharray.roaringish.roaringish import RoaringishEncoder

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
#     ----
#
#     Finally: X   X   X    X     <- 4 words selected from each to find max span
#
#
def _intersect_all(posns_encoded: List[np.ndarray]):
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
    # Merge all the rest to grab them
    int_header = encoder.header(lhs_int)
    to_left_header = encoder.header(lhs_to_left)
    to_right_header = encoder.header(lhs_to_right)
    merged = merge(int_header, to_left_header, drop_duplicates=True)
    merged = merge(merged, to_right_header, drop_duplicates=True)

    # Slice only matches at MSBs
    for i in range(len(posns_encoded)):
        posns_encoded[i] = encoder.slice(posns_encoded[i], merged)


#     Picking up from intersections:
#
#     Finally: X   X   X    X    X  X  <- 6 words selected from each to find max span
#
#     Each X may be adjacent or not
#
#     Looking at LSBs:
#        MSB:     9    10         13    14    15         20
#     Term A  01001 00000      00000 01000 00001      00100
#     Term B  00010 00010      00010 00000 01000      00010
#     Term C  10000 00001      00000 00001 00000      00001
#
#     Limit to two adj words at a time (slop must be < LSB bits)
#
#        MSB:     9    10         13    14    15         20
#     Term A  01001 00000      00000 01000 00001      00100
#     Term B  00010 00010      00010 00000 01000      00010
#     Term C  10000 00001      00000 00001 00000      00001
#             -----------
#             (has span?)
#                              -----------
#                              (has span?)
#                                     -----------
#                                     (has span?)
#                                                     -----
#                                                     (has span?)
#
#     Has span algo, really finding minimum sized spans
#
#       Concat to single 64 bit word
#            0100100000
#            0001000010
#            1000000001
#
#     Search w/ mask
#            ffff000000
#     And'd, now check if > 0:
#            0100000000 > 0 -> 1      0      0
#            0001000000 > 0 -> 1      1      0
#            1000000000 > 0 -> 1      1      0
#
#     has_span (ANDED):        1
#
#     Can we shrink the mask?
#     If any hase leading or trailing 0s in mask, then shrink
def span_search(posns_encoded: List[np.ndarray],
                phrase_freqs: np.ndarray,
                slop: int) -> np.ndarray:
    """Find span matches up to PAYLOAD_LSB bits span distance."""
    # Find inner span candidates
    _intersect_all(posns_encoded)

    # Populate phrase freqs with matches of slop
    r_span_search(posns_encoded, phrase_freqs, slop,
                  encoder.key_mask,
                  encoder.header_mask,
                  encoder.payload_lsb_bits)
    return phrase_freqs
