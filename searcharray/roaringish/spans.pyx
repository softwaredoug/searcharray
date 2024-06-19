# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
"""Utilities for computing spans for position aware search with slop > 0."""
cimport numpy as np
import numpy as np

cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport DTYPE_t


cdef extern from "stddef.h":
    # Trailing and leading zeros to trim the span mask
    int __builtin_popcountll(unsigned long long x)
    int __builtin_ctzll(unsigned long long x)
    int __builtin_clzll(unsigned long long x)


cdef _count_spans_of_slop(DTYPE_t[:] posns, DTYPE_t slop):
    # DTYPE_t mask = 0xFFFFFFFFFFFFFFFF
    for posn in posns:
        if posn == 0:
            return 0
    return 1


cdef _get_adj_spans(DTYPE_t[:, :] posns_arr,
                    DTYPE_t[:] phrase_freqs,
                    DTYPE_t slop,
                    DTYPE_t key_mask,
                    DTYPE_t header_mask,
                    DTYPE_t lsb_bits):
    """Get unscored spans."""
    pass


cdef _span_freqs(DTYPE_t[:, :] posns_arr,
                 double[:] phrase_freqs,
                 DTYPE_t slop,
                 DTYPE_t key_mask,
                 DTYPE_t header_mask,
                 DTYPE_t lsb_bits):
    """Get unscored spans, within 64 bits."""

    cdef DTYPE_t i = 0
    cdef DTYPE_t j = 0
    cdef DTYPE_t k = 0
    cdef DTYPE_t adj = 0
    cdef DTYPE_t set_idx = 0
    cdef DTYPE_t curr_msb = 0
    cdef DTYPE_t posn = 0
    cdef DTYPE_t payload_mask = ~header_mask
    cdef unsigned char[:] which_terms = -np.ones(lsb_bits * 2, dtype=np.uint8)
    # Assuming no overlaps.
    #
    # Collect the term where each position is set
    #
    #        term1: 010011010000        term1 & (term2 + 1)
    #        term2: 000000000001
    #        term3: 000000000010
    #
    # which_terms=  F0FF00F0FF21
    # (really which_terms [last_posns] [this_posns])
    #
    # Scan the which_terms to find spans within slop
    # Remove them, increment the phrase_freqs for the doc, then continue
    # It may seem we can scan which_terms, but we can just get the minimum spans
    #
    # which_terms=  F0FF00F0FF21
    #
    # Then diffs:
    # which_terms=  F0FF00F0FF21
    #       dist    001201201245
    #      coll?               *       <- when to collect, every prev seen unique num
    #
    # which_terms = F12F00F0FF21
    #       dist    011230101223
    #      coll?        *      *
    # This one is tricky because we should NOT collect the first time we encounter all
    # terms, but rather the min span in between
    #
    # which_terms = F2110120FF21
    #       dist    0234500
    #      coll?          *     *
    #
    # which_terms = F2110120FF21
    #       spans    ----             1 (posn_first_term, posn_last_term, terms_enc, span_score)
    #                 -----           2 (posn_first_term, posn_last_term, terms_enc, span_score)
    #                  ----
    #                   ---
    #                      -----
    #
    #  We have to track all active spans
    #   when popcount terms_enc = num_terms
    #      ... we collect the span
    #   if overlaps and size smaller than existing collected span
    #      remove the existing span
    #
    #   span score is the current span slop
    #
    #  Now we have spans
    #
    # which_terms = F2110120FF21
    #                   ---
    #                      -----
    #
    #
    #
    # Now we score the span to see if its < slop
    #
    # curr_posns are current bits analyzed for slop
    cdef np.uint64_t[:] curr_posns = np.empty(posns_arr.shape[0], dtype=np.uint64)
    for i in range(posns_arr.shape[1]):

        # Each term
        for j in range(posns_arr.shape[0]):
            # Each msb
            # Later optimization - could we do this without storing which_terms?
            term = posns_arr[j, i] & payload_mask
            set_idx = __builtin_ctzll(term)
            posns_arr[j, i] &= ~(1 << set_idx)
            which_terms[set_idx] = j

        # Gather and score min spans
        for posn in which_terms:
            print(posn)
            if posn == 0xFF:
                continue

        # Shift the which_terms up by num_payload_bits
        for j in range(64 - lsb_bits):
            which_terms[j + lsb_bits] = which_terms[j]

        # The min popcount is the upper bound of phrase freq
        # popcount_xored_min = 128
        # for j in range(posns_arr.shape[0]):
        #     popcount_xored = __builtin_popcountll(posns_arr[j, i] ^ max_span_mask)
        #     if popcount_xored < popcount_xored_min:
        #         popcount_xored_min = popcount_xored


def span_search(np.ndarray[DTYPE_t, ndim=2] posns_arr,
                np.ndarray[double, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t lsb_bits):
    _span_freqs(posns_arr, phrase_freqs, slop, key_mask, header_mask, lsb_bits)
