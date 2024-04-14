"""Utilities for computing spans for position aware search with slop > 0."""
cimport numpy as np
import numpy as np
from enum import Enum

cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t


# cdef extern from "stddef.h":
    # Get ctz and clz
#     int __builtin_ctzll(unsigned long long x)
 #    int __builtin_clzll(unsigned long long x)


cdef _count_spans_of_slop(DTYPE_t[:] posns, DTYPE_t slop):
    # DTYPE_t mask = 0xFFFFFFFFFFFFFFFF
    for posn in posns:
        if posn == 0:
            return 0
    return 1


cdef _span_search(DTYPE_t[:, :] posns_arr, 
                  DTYPE_t[:] phrase_freqs,
                  DTYPE_t slop,
                  DTYPE_t key_mask,
                  DTYPE_t header_mask, 
                  DTYPE_t lsb_bits):

    cdef DTYPE_t i = 0
    cdef DTYPE_t j = 0
    cdef DTYPE_t adj = 0
    cdef DTYPE_t curr_msb = 0
    cdef DTYPE_t adj_msb = 0
    # curr_posns are current bits analyzed for slop
    cdef np.uint64_t[:] curr_posns = np.empty(posns_arr.shape[0], dtype=np.uint64)
    for i in range(posns_arr.shape[1]):

        # First get self + adj into a single 64 bit number
        # per term
        #         i:    adj:
        #         10    11         14
        # termA   0011  0001       0010 
        # termB   0100  1000       0001
        #
        # curr_posns now:
        #
        # termA:  [00110001,
        # termB:   00100000]
        #
        # Now we can check for minspans in each terms words
        # 
        doc_id = posns_arr[0, i] & key_mask
        for j in range(posns_arr.shape[0]):
            curr_posns[j] = posns_arr[j, i] << lsb_bits

        adj = i + 1
        if adj < posns_arr.shape[1]:
            adj_msb = posns_arr[i, 0] & header_mask
            adj_msb += (1 << lsb_bits)
            curr_msb = posns_arr[adj, 0] & header_mask

            # If my neighbor is actually a neighbor
            if curr_msb == adj_msb:
                for j in range(posns_arr.shape[0]):
                    curr_posns[j] |= posns_arr[j, adj]
        # Find a min span
        phrase_freqs[doc_id] = _count_spans_of_slop(curr_posns, slop)


def span_search(np.ndarray[DTYPE_t, ndim=2] posns_arr,
                np.ndarray[DTYPE_t, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t lsb_bits):
    _span_search(posns_arr, phrase_freqs, slop, key_mask, header_mask, lsb_bits)
