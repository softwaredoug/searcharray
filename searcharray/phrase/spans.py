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
#     term C        | 11             | 14
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
#     Intersecting
#     term A     |10|       | 12|  13                18
#     term B     |10|       | 12|       14   15 17
#     Adjacent
#     term A      10          12|  13|               18
#     term B      10          12|    |  14   15 17 |
#        Giving 9.5, 10.5  11.5, 12.5, 13.5    , 17.5
#     ------
#     Next pair, intersecting
#     term B      10          12      | 14 |   15 17       * RHS intersect -> 13 -> 14
#     term C          11              | 14 |
#     Adjacent
#     term B      10          12        14     15 17
#     term C         |11                14
#        Giving 10.5, 13.5, 14.5
#     w/ above  10.5, 13.5
#     ------
#
#
#     Finally: X   X   X    X     <- 4 words selected from each to find max span
#
# Does it work to compute any intersection, and then attach any adjecent that exist
#
#

def hdr(arr):
    return encoder.header(arr) >> (_64 - encoder.header_bits)


def _intersect_all(posns_encoded: List[np.ndarray]):
    """Intersect all encoded positions at roaringish MSBs."""
    if len(posns_encoded) < 2:
        raise ValueError("Need at least two positions to intersect")
    last_lhs_headers = None
    last_rhs_headers = None
    curr = posns_encoded[0]
    for term_idx, posns_next in enumerate(posns_encoded[1:]):
        lhs_int_idx, rhs_int = intersect(curr, posns_next, mask=encoder.header_mask)
        int_headers = encoder.header(curr[lhs_int_idx])

        # What is adjacent on LHS / RHS of interserction
        # Next to left
        # 0 ->           2
        #     <-  1   (curr to right)
        #         1 ->  (keep
        # 0           <-   2
        curr_to_right, next_to_left = adjacent(curr, posns_next, mask=encoder.header_mask)
        lhs_headers = merge(int_headers, posns_next[next_to_left])
        rhs_headers = merge(int_headers, curr[curr_to_right])
        next_to_right, curr_to_left = adjacent(posns_next, curr, mask=encoder.header_mask)
        lhs_headers = merge(lhs_headers, curr[curr_to_left])
        rhs_headers = merge(rhs_headers, posns_next[next_to_right])

        if last_lhs_headers is not None:
            lhs, _ = intersect(last_lhs_headers, lhs_headers, mask=encoder.header_mask)
            rhs, _ = intersect(last_rhs_headers, rhs_headers, mask=encoder.header_mask)
            last_lhs_headers = last_lhs_headers[lhs]
            last_rhs_headers = last_rhs_headers[rhs]
        else:
            last_lhs_headers = lhs_headers
            last_rhs_headers = rhs_headers

        # Update by intersecting with current working lhs / rhs headers

    assert last_rhs_headers is not None
    assert last_lhs_headers is not None
    to_rhs = last_rhs_headers + (_1 << (_64 - encoder.header_bits))
    to_lhs = last_lhs_headers - (_1 << (_64 - encoder.header_bits))
    all_headers = merge(to_rhs, to_lhs, drop_duplicates=True)
    all_headers = merge(last_lhs_headers, all_headers, drop_duplicates=True)
    all_headers = merge(last_rhs_headers, all_headers, drop_duplicates=True)
    all_headers = all_headers & encoder.header_mask
    # Get active MSBs now
    # Merge all the rest to grab them

    # Slice only matches at header MSBs
    new_posns_encoded = posns_encoded.copy()
    for i in range(len(posns_encoded)):
        new_posns_encoded[i] = encoder.slice(posns_encoded[i], header=all_headers)
    lengths = np.cumsum([0] + [len(posns) for posns in new_posns_encoded], dtype=np.uint64)
    concatted = np.concatenate(new_posns_encoded, dtype=np.uint64)
    return concatted, lengths


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
    # Find posns to check for span candidates
    posns, lengths = _intersect_all(posns_encoded)

    # Populate phrase freqs with matches of slop
    r_span_search(posns, lengths,
                  phrase_freqs, slop,
                  encoder.key_mask,
                  encoder.header_mask,
                  encoder.key_bits,
                  encoder.payload_lsb_bits)
    return phrase_freqs
