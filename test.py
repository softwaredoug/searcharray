import numpy as np
from searcharray.roaringish import popcount64, msb_mask64
#
#              streak each some given slop amount
# 001100       self >> 2 | self >> 1 | self | self << 1 | self << 2 | .. | self << n
# 000010
# 000001
# Expected:
# 000111

# 111100
# 000000
# 000001
# Expected:
# 000101

# 111100
# 000010
# 000001
# Expected:
# 000111
#
# 000001
# 000010
# 000100
# Expected:
# 000111

# 111100
# 000010
# Expected:
# 000110

# 100000
# 010000
# 001000
# 000100
# 000010
# 000001
# Expected:
# 111111

# ------000000
# 100100000000
# 010000000001
# 001000000010
# 000100000000
# 000010000001
# 000001000000  (streak phrase size bits + slop)
# Expected:
# 111111

#
# How would you adapt this to span across multiple values?
#
# 100 100000000
# 010 000000001
# 001 000000010
# 000 100000000
# 000 010000001
# 000 001000000  (streak phrase size bits + slop)
# Expected:
# 111 111
#
# 1001 00000000
# 0100 00000001
# 0010 00000010
# 0001 00000000
# 0000 10000001
# 0000 01000000  (streak phrase size bits + slop)
# Expected:
# 1111 11
# If there are N
#
#
# One option, just take the candidate mask and concat from each side
#
# mask = (lhs_mask, adj_bits)
#
# Get a mask of adj_bits num bits:
# mask_of_adj_bit_len = (1 << adj_bits) - 1
#
# Shift to upper 64 bits:
# rhs_mask = mask_of_adj_bit_len << 64 - adj_bits
# values = (lhs & lhs_mask << adj_bits) | (rhs & rhs_mask)
#
# Run the normal argorithm on values
#  1. all bits must be set
#  2. shrink span based on leading / trailing zeros
#
#
# lhs &


_1 = np.uint64(1)


def u64(v):
    return np.uint64(v)


def i64(v):
    return np.int64(v)


# 00001011
#     ----
#
# 11110100 - 100 -> 11110000 - 100 -> 11110000
# 00001111  <-- this feels closer!

# (Pdb) dump( (u64(~mask >> u64(3)) & mask) & (u64(~mask >> u64(4)) & mask) & (u64(~mask >> u64

def dump(v):
    if isinstance(v, np.ndarray):
        print([f"{x:032b}" for x in v])
    else:
        print(f"{v:0b}")


def popcount_min(bit_values):
    popcounts = popcount64(bit_values)
    return np.min(popcounts)


def lsb(val):
    return u64(val & u64(-i64(val)))


def msb(val):
    return msb_mask64(np.asarray([val], dtype=np.uint64))[0]


def spans_within_mask(bit_vals, mask):
    masked = bit_vals & mask
    most_matches = popcount_min(masked)
    # MSB or LSB of mask is zero of all, we can chop
    # off from mask next time where its 0
    # NOTE we could use ctz or clz to count trailing or leading
    # zeros in one CPU instruction
    shrinks_lsb = np.bitwise_and.reduce(~masked) & lsb(mask) > 0
    shrinks_msb = np.bitwise_and.reduce(~masked) & msb(mask) > 0
    return most_matches, shrinks_lsb, shrinks_msb


def naive_max_span(bit_vals):
    """Closest sets of bits for each bit val."""
    # Create a mask of len(bit_vals) + slop
    mask = np.bitwise_xor.reduce(bit_vals)
    # For every such mask, can we find smallest place where each has bits set
    # This can be a one step ctz / clz
    while True:
        num_candidates, shrinks_lsb, shrinks_msb = spans_within_mask(bit_vals, mask)
        # TODO when num_candidates > 1, split mask and do two searches
        # by finding where all values have a 0 mask & ~(bitvals anded)
        if num_candidates == 0:
            return 0
        if mask == 0:
            break
        if shrinks_lsb:
            mask &= ~lsb(mask)
        elif shrinks_msb:
            mask &= ~msb(mask)
        else:
            print(num_candidates)
            dump(mask)
            break
    return mask
    # Within this there exists


if __name__ == "__main__":
    arr = np.asarray([0b1000000100,
                      0b0000001000,
                      0b0010110000], dtype=np.uint64)
    naive_max_span(arr)
    assert naive_max_span(arr) & u64(0b0000111000) == u64(0b0000111000)
    arr = np.asarray([0b1111100000,
                      0b0000001000,
                      0b0000010000], dtype=np.uint64)
    assert naive_max_span(arr) & u64(0b0000111000) == u64(0b0000111000)
    arr = np.asarray([0b100000,
                      0b000000,
                      0b000100], dtype=np.uint64)
    assert naive_max_span(arr) == 0
    arr = np.asarray([0b001100,
                      0b000010,
                      0b000001], dtype=np.uint64)
    assert naive_max_span(arr) & u64(0b000111) == u64(0b000111)
    arr = np.asarray([0b110000,
                      0b001000,
                      0b000100], dtype=np.uint64)
    assert naive_max_span(arr) & u64(0b011100) == u64(0b011100)
    arr = np.asarray([0b110000,
                      0b001000,
                      0b000100], dtype=np.uint64)
    assert naive_max_span(arr) & u64(0b011100) == u64(0b011100)
    arr = np.asarray([0b100000,
                      0b100000,
                      0b010000], dtype=np.uint64)
    assert naive_max_span(arr) == 0
