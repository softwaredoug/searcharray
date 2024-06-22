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


cdef struct ActiveSpans:
    DTYPE_t[64] terms
    DTYPE_t[64] posns
    np.int64_t[64] beg
    np.int64_t[64] end
    np.uint64_t cursor


cdef ActiveSpans _new_active_spans():
    cdef ActiveSpans active_spans
    active_spans.terms = np.zeros(64, dtype=np.uint64)
    active_spans.posns = np.zeros(64, dtype=np.uint64)
    active_spans.beg = np.zeros(64, dtype=np.int64)
    active_spans.end = np.zeros(64, dtype=np.int64)
    active_spans.cursor = 0
    return active_spans


cdef void _clear_span(ActiveSpans* spans, DTYPE_t span_idx):
    spans.terms[span_idx] = 0
    spans.posns[span_idx] = 0
    spans.beg[span_idx] = 0
    spans.end[span_idx] = 0


cdef np.int64_t _span_width(ActiveSpans* spans, DTYPE_t span_idx):
    return abs(spans.end[span_idx] - spans.beg[span_idx])


cdef DTYPE_t _consume_lsb(DTYPE_t* term):
    """Get lowest set bit, clear it, and return the position it occured in."""
    lsb = __builtin_ctzll(term[0])
    # Clear LSB
    term[0] = (term[0] & (term[0] - 1))
    return lsb


cdef DTYPE_t _posn_mask(DTYPE_t set_idx, DTYPE_t payload_base):
    return 1 << ((set_idx + payload_base) % 64)


cdef DTYPE_t _num_terms(ActiveSpans* spans, DTYPE_t span_idx):
    return __builtin_popcountll(spans.terms[span_idx])


cdef DTYPE_t _num_posns(ActiveSpans* spans, DTYPE_t span_idx):
    return __builtin_popcountll(spans.posns[span_idx])


cdef bint _do_spans_overlap(ActiveSpans* spans, DTYPE_t span_idx_lhs, DTYPE_t span_idx_rhs):
    return (spans.beg[span_idx_lhs] <= spans.end[span_idx_rhs]) and (spans.end[span_idx_lhs] >= spans.beg[span_idx_rhs])


cdef bint _is_span_complete(ActiveSpans* spans, DTYPE_t span_idx, DTYPE_t num_terms):
    cdef DTYPE_t num_terms_visited = _num_terms(spans, span_idx)
    cdef DTYPE_t num_posns_visited = _num_posns(spans, span_idx)
    return (num_terms_visited == num_terms) or (num_posns_visited == num_terms)



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

    cdef ActiveSpans spans = _new_active_spans()

    cdef np.uint64_t[:] curr_idx = np.zeros(64, dtype=np.uint64)
    cdef np.uint64_t curr_term_mask = 0
    cdef np.uint64_t posn_mask = 0
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
                curr_term_mask = 0x1 << term_ord

                # Consume every position into every possible span
                while term != 0:
                    set_idx = _consume_lsb(&term)
                    posn_mask = _posn_mask(set_idx, payload_base)

                    spans.terms[spans.cursor] = curr_term_mask
                    spans.posns[spans.cursor] = posn_mask
                    if term_ord == 0:
                        spans.beg[spans.cursor] = set_idx

                    # Remove spans that are too long
                    for span_idx in range(spans.cursor):
                        # Continue active spans
                        num_terms_visited = _num_terms(&spans, span_idx)
                        num_posns_visited = _num_posns(&spans, span_idx)
                        if num_terms_visited < num_terms and num_posns_visited == num_terms:
                            continue
                        spans.terms[span_idx] |= curr_term_mask
                        num_terms_visited_now = _num_terms(&spans, span_idx)
                        if num_terms_visited_now > num_terms_visited:
                            # Add position for new unique term
                            spans.posns[span_idx] |= posn_mask
                            new_unique_posns = _num_posns(&spans, span_idx)
                            if num_posns_visited == new_unique_posns:
                                # Clear curr_term_mask and cancel this position, we've seen it before
                                spans.terms[span_idx] &= ~curr_term_mask
                                continue
                            spans.end[span_idx] = set_idx
                        # If all terms visited, see if we should remove
                        if (num_terms_visited_now == num_terms) \
                           and _span_width(&spans, span_idx) > num_terms + slop:
                            _clear_span(&spans, span_idx)

                    if spans.cursor > 64:
                        break
                    spans.cursor += 1
                    last_set_idx = set_idx
                curr_idx[term_ord] += 1
                if curr_idx[term_ord] < lengths[term_ord+1]:
                    curr_key = (posns[curr_idx[term_ord]] & key_mask) >> (64 - key_bits)
                payload_base += lsb_bits
                if curr_key != last_key or spans.cursor > 64:
                    payload_base = 0
                    break

        # All terms consumed for doc

        # Count phrase freqs
        for span_idx in range(spans.cursor):
            if not _is_span_complete(&spans, span_idx, num_terms):
                continue
            for other_span_idx in range(spans.cursor):
                if other_span_idx == span_idx or not _is_span_complete(&spans, other_span_idx, num_terms):
                    continue
                if _do_spans_overlap(&spans, span_idx, other_span_idx):
                    break
            assert last_key < phrase_freqs.shape[0]
            phrase_freqs[last_key] += 1

        # Reset
        spans = _new_active_spans()


def span_search(np.ndarray[DTYPE_t, ndim=1] posns,
                np.ndarray[DTYPE_t, ndim=1] lengths,
                np.ndarray[double, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t key_bits,
                DTYPE_t lsb_bits):
    _span_freqs(posns, lengths, phrase_freqs, slop, key_mask, header_mask, key_bits, lsb_bits)
