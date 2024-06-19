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


cdef _bits_set(DTYPE_t value,
               DTYPE_t[:] buffer64,
               DTYPE_t* buffer_write_len):
    """Get the bits set in a 64 bit value."""
    cdef DTYPE_t lsb = 0
    cdef DTYPE_t bit_posn = 0
    buffer_write_len[0] = 0
    while value > 0:
        lsb = value & -value
        bit_posn = __builtin_ctzll(lsb)
        print(f"LSB: {lsb:064b} | bt_posn: {bit_posn}")
        buffer64[buffer_write_len[0]] = bit_posn
        # Clear LSB
        value &= value - 1
        buffer_write_len[0] += 1


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
    cdef np.uint64_t[:] bits_set = np.empty(64, dtype=np.uint64)
    cdef np.uint64_t bits_set_len = 0

    cdef np.uint64_t[:] active_spans_queue = np.empty(64, dtype=np.uint64)
    cdef np.int64_t[:] span_score_queue = np.empty(64, dtype=np.int64)
    cdef np.uint64_t next_active_beg = 0
    cdef np.uint64_t curr_term_mask = 0
    cdef np.uint64_t num_terms = posns_arr.shape[0]
    cdef np.uint64_t all_terms_mask = (1 << num_terms) - 1
    cdef np.uint64_t term_ord = 0
    cdef np.uint64_t curr_key = 0
    cdef np.uint64_t last_key = 0
    cdef np.uint64_t payload_base = 0
    last_set_idx = 0
    for i in range(posns_arr.shape[1]):
        curr_key = posns_arr[0, i] & key_mask

        if curr_key != last_key:
            print("-----------")
            print(f"Collecting spans for {curr_key} - {next_active_beg} active spans")

            # Make new active span queue
            new_active_span_queue = np.empty(64, dtype=np.uint64)
            new_span_score_queue = np.empty(64, dtype=np.int64)

            # Copy existing
            for span_idx in range(next_active_beg):
                span_size = __builtin_popcountll(active_spans_queue[span_idx])
                print(f"Span {span_idx} size: {span_size}")
                if span_size != num_terms:
                    continue
                print("Keeping span")
                print("Span score: ", span_score_queue[span_idx])
                phrase_freqs[curr_key] += 1
            
            next_active_beg = 0
            active_spans_queue = new_active_span_queue
            span_score_queue = new_span_score_queue

        # We may not be processing each term in order
        # Each term is potentially a new span
        for term_ord in range(num_terms):
            # Each msb
            term = posns_arr[term_ord, i] & payload_mask
            _bits_set(term, bits_set, &bits_set_len)
            last_set_idx = 0
            print("---")
            print(f"Term ord: {term_ord}, bits_set_len: {bits_set_len}")
            for idx in range(bits_set_len):
                set_idx = bits_set[idx]
                print(f"set_idx: {set_idx}, last_set_idx: {last_set_idx}, payload_base: {payload_base}")
                curr_term_mask = 0x1 << term_ord
                active_spans_queue[next_active_beg] = curr_term_mask
                span_score_queue[next_active_beg] = term_ord   # The term index as start score, because 0 is in order
                for span_idx in range(next_active_beg):
                    if __builtin_popcountll(active_spans_queue[span_idx]) == num_terms:
                        print(f" Not updating completed span {span_idx} score: {span_score_queue[span_idx]}")
                        continue
                    active_spans_queue[span_idx] |= curr_term_mask
                    span_score_queue[span_idx] += set_idx - last_set_idx - 1
                    if span_score_queue[span_idx] > slop:
                        print(f"Removing span {span_idx} w/ score {span_score_queue[span_idx]}")
                        active_spans_queue[span_idx] = 0
                        span_score_queue[span_idx] = 0x7FFFFFFFFFFFFFFF
                    else:
                        print(f" Keeping Span {span_idx} score: {span_score_queue[span_idx]}")
                next_active_beg += 1
                last_set_idx = set_idx

        payload_base += lsb_bits
        last_key = curr_key

    # Copy existing
    print("-----------")
    print(f"Collecting spans for {curr_key} - {next_active_beg} active spans")
    for span_idx in range(next_active_beg):
        span_size = __builtin_popcountll(active_spans_queue[span_idx])
        print(f"Span {span_idx} size: {span_size}")
        if span_size != num_terms:
            continue
        print("Keeping span")
        print("Span score: ", span_score_queue[span_idx])
        phrase_freqs[curr_key] += 1


def span_search(np.ndarray[DTYPE_t, ndim=2] posns_arr,
                np.ndarray[double, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t lsb_bits):
    _span_freqs(posns_arr, phrase_freqs, slop, key_mask, header_mask, lsb_bits)
