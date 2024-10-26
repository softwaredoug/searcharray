"""Naive popcount implementation until such time that's exposed in numpy (SOON!)."""
import numpy as np


m1 = np.uint64(0x5555555555555555)
m2 = np.uint64(0x3333333333333333)
m3 = np.uint64(0x0F0F0F0F0F0F0F0F)
m4 = np.uint64(0x0101010101010101)


mask = np.int64(-1).view(np.uint64)
s55 = np.uint64(m1 & mask)  # Add more digits for 128bit support
s33 = np.uint64(m2 & mask)
s0F = np.uint64(m3 & mask)
s01 = np.uint64(m4 & mask)
num_bytes_64 = 8
all_but_one_bit = np.uint64(8 * (num_bytes_64 - 1))

_1 = np.uint64(1)
_2 = np.uint64(2)
_4 = np.uint64(4)


def bit_count64(arr):
    """Count the number of bits set in each element in the array."""
    arr = arr - ((arr >> _1) & s55)
    arr = (arr & s33) + ((arr >> _2) & s33)

    arr += (arr >> _4)
    arr &= s0F
    arr *= s01
    arr >>= all_but_one_bit

    return arr
