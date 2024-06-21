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


cdef _span_freqs(DTYPE_t[:] posns,      # Flattened all terms in one array
                 DTYPE_t[:] lengths,
                 double[:] phrase_freqs,
                 DTYPE_t slop,
                 DTYPE_t key_mask,
                 DTYPE_t header_mask,
                 DTYPE_t key_bits,
                 DTYPE_t lsb_bits):
    """Get unscored spans, within 64 bits."""

    cdef DTYPE_t set_idx = 0
    cdef DTYPE_t payload_mask = ~header_mask
    cdef np.uint64_t[:] curr_idx = np.zeros(64, dtype=np.uint64)
    cdef np.uint64_t[:] active_spans_queue = np.zeros(64, dtype=np.uint64)
    cdef np.uint64_t[:] active_spans_posns = np.zeros(64, dtype=np.uint64)
    cdef np.int64_t[:] span_beg = np.empty(64, dtype=np.int64)
    cdef np.int64_t[:] span_end = np.empty(64, dtype=np.int64)
    cdef np.uint64_t next_active_beg = 0
    cdef np.uint64_t curr_term_mask = 0
    cdef np.uint64_t num_terms = len(lengths) - 1
    cdef np.uint64_t term_ord = 0
    cdef np.uint64_t curr_key = 0
    cdef np.uint64_t last_key = 0
    cdef np.uint64_t payload_base = 0
    last_set_idx = 0

    for i in range(num_terms):
        curr_idx[i] = lengths[i]

    while curr_idx[0] < lengths[1]:
        # Read each term up to the next  doc
        last_key = -1
        for term_ord in range(num_terms):
            curr_key = ((posns[curr_idx[term_ord]] & key_mask) >> (64 - key_bits))
            payload_base = 0
            while curr_idx[term_ord] < lengths[term_ord+1]:
                last_key = curr_key
                term = posns[curr_idx[term_ord]] & payload_mask

                while term != 0:
                    # Consume into span
                    set_idx = __builtin_ctzll(term)
                    # Clear LSB
                    term = (term & (term - 1))
                    # Start a span
                    curr_term_mask = 0x1 << term_ord
                    active_spans_queue[next_active_beg] = curr_term_mask
                    active_spans_posns[next_active_beg] = 1 << ((set_idx + payload_base) % 64)
                    if term_ord == 0:
                        span_beg[next_active_beg] = set_idx

                    # Remove spans that are too long
                    for span_idx in range(next_active_beg):
                        # Continue active spans
                        num_terms_visited = __builtin_popcountll(active_spans_queue[span_idx])
                        num_posns_visited = __builtin_popcountll(active_spans_posns[span_idx])
                        if num_terms_visited < num_terms and num_posns_visited == num_terms:
                            continue
                        active_spans_queue[span_idx] |= curr_term_mask
                        num_terms_visited_now = __builtin_popcountll(active_spans_queue[span_idx])
                        if num_terms_visited_now > num_terms_visited:
                            # Add position for new unique term
                            num_unique_posns = __builtin_popcountll(active_spans_posns[span_idx])
                            active_spans_posns[span_idx] |= 1 << ((set_idx + payload_base) % 64)
                            new_unique_posns = __builtin_popcountll(active_spans_posns[span_idx])
                            if num_unique_posns == new_unique_posns:
                                # Clear curr_term_mask and cancel this position, we've seen it before
                                active_spans_queue[span_idx] &= ~curr_term_mask
                                continue
                            span_end[span_idx] = set_idx
                        # If all terms visited, see if we should remove
                        if (num_terms_visited_now == num_terms) \
                           and abs(span_end[span_idx] - span_beg[span_idx]) > num_terms + slop:
                            span_beg[span_idx] = 0
                            span_end[span_idx] = 0
                            active_spans_queue[span_idx] = 0
                            active_spans_posns[span_idx] = 0

                    if next_active_beg > 64:
                        break
                    next_active_beg += 1
                    last_set_idx = set_idx
                curr_idx[term_ord] += 1
                curr_key = (posns[curr_idx[term_ord]] & key_mask) >> (64 - key_bits)
                payload_base += lsb_bits
                if curr_key != last_key or next_active_beg > 64:
                    payload_base = 0
                    break

        # All terms consumed for doc

        # Count phrase freqs
        for span_idx in range(next_active_beg):
            num_terms_visited = __builtin_popcountll(active_spans_queue[span_idx])
            num_posns_visited = __builtin_popcountll(active_spans_posns[span_idx])
            if num_terms_visited < num_terms or num_posns_visited < num_terms:
                continue
            phrase_freqs[last_key] += 1

        # Reset
        next_active_beg = 0
        active_spans_queue = np.zeros(64, dtype=np.uint64)
        active_spans_posns = np.zeros(64, dtype=np.uint64)
        span_beg = np.zeros(64, dtype=np.int64)
        span_end = np.zeros(64, dtype=np.int64)


def span_search(np.ndarray[DTYPE_t, ndim=1] posns,
                np.ndarray[DTYPE_t, ndim=1] lengths,
                np.ndarray[double, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t key_bits,
                DTYPE_t lsb_bits):
    _span_freqs(posns, lengths, phrase_freqs, slop, key_mask, header_mask, key_bits, lsb_bits)
