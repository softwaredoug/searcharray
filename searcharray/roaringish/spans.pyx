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
    DTYPE_t[512] terms
    DTYPE_t[512] posns
    np.int64_t[512] beg
    np.int64_t[512] end
    np.uint64_t cursor


cdef ActiveSpans _new_active_spans():
    cdef ActiveSpans active_spans
    active_spans.terms = np.zeros(512, dtype=np.uint64)
    active_spans.posns = np.zeros(512, dtype=np.uint64)
    active_spans.beg = np.zeros(512, dtype=np.int64)
    active_spans.end = np.zeros(512, dtype=np.int64)
    active_spans.cursor = 0
    return active_spans


cdef void _clear_span(ActiveSpans* spans, DTYPE_t span_idx):
    spans[0].terms[span_idx] = 0
    spans[0].posns[span_idx] = 0
    spans[0].beg[span_idx] = 0
    spans[0].end[span_idx] = 0


cdef np.int64_t _span_width(ActiveSpans* spans, DTYPE_t span_idx):
    return abs(spans[0].end[span_idx] - spans[0].beg[span_idx])


cdef DTYPE_t _consume_lsb(DTYPE_t* term):
    """Get lowest set bit, clear it, and return the position it occured in."""
    lsb = __builtin_ctzll(term[0])
    # Clear LSB
    term[0] = (term[0] & (term[0] - 1))
    return lsb


cdef DTYPE_t _posn_mask(np.int64_t curr_posn):
    return 1 << (curr_posn % 64)


cdef DTYPE_t _num_terms(ActiveSpans* spans, DTYPE_t span_idx):
    return __builtin_popcountll(spans[0].terms[span_idx])


cdef DTYPE_t _num_posns(ActiveSpans* spans, DTYPE_t span_idx):
    return __builtin_popcountll(spans[0].posns[span_idx])


cdef bint _do_spans_overlap(ActiveSpans* spans_lhs, DTYPE_t span_idx_lhs,
                            ActiveSpans* spans_rhs, DTYPE_t span_idx_rhs):
    return ((spans_lhs.beg[span_idx_lhs] <= spans_rhs.end[span_idx_rhs])
            and (spans_lhs.end[span_idx_lhs] >= spans_rhs.beg[span_idx_rhs]))


cdef bint _is_span_complete(ActiveSpans* spans, DTYPE_t span_idx, DTYPE_t num_terms):
    cdef DTYPE_t num_terms_visited = _num_terms(spans, span_idx)
    cdef DTYPE_t num_posns_visited = _num_posns(spans, span_idx)
    return (num_terms_visited == num_terms) or (num_posns_visited == num_terms)


cdef bint _is_span_unsalvagable(ActiveSpans* spans, DTYPE_t span_idx, DTYPE_t max_width):
    return (spans[0].beg[span_idx] < spans[0].end[span_idx]) and (_span_width(spans, span_idx) > max_width)


cdef void _print_span(ActiveSpans* spans, DTYPE_t span_idx):
    cdef np.uint64_t terms = spans[0].terms[span_idx]
    cdef np.uint64_t posns = spans[0].posns[span_idx]
    print(f"{span_idx}: term:{terms:b} posns:{posns:b} beg-end:{spans[0].beg[span_idx]}-{spans[0].end[span_idx]}")


cdef ActiveSpans _compact_spans(ActiveSpans* spans, DTYPE_t max_width):
    """Copy only active spans into new spans."""
    cdef ActiveSpans new_spans = _new_active_spans()
    cdef span_idx = 0
    for span_idx in range(spans[0].cursor):
        if _span_width(spans, span_idx) > max_width:
            continue
        if _num_terms(spans, span_idx) > 0:
            new_spans.terms[new_spans.cursor] = spans[0].terms[span_idx]
            new_spans.posns[new_spans.cursor] = spans[0].posns[span_idx]
            new_spans.beg[new_spans.cursor] = spans[0].beg[span_idx]
            new_spans.end[new_spans.cursor] = spans[0].end[span_idx]
            new_spans.cursor += 1
    return new_spans


cdef ActiveSpans _collect_spans(ActiveSpans* spans, DTYPE_t num_terms, DTYPE_t max_width):
    """Sort so shortest spans are first."""
    # TODO - if spans were a heap, this might be faster
    cdef span_idx = 0
    cdef coll_span_idx = 0
    cdef bint overlaps = False
    cdef ActiveSpans collected_spans = _new_active_spans()
    for span_idx in range(spans[0].cursor):
        if _is_span_complete(spans, span_idx, num_terms) and _span_width(spans, span_idx) < max_width:
            new_width = abs(spans[0].end[span_idx] - spans[0].beg[span_idx])
            overlaps = False
            for coll_span_idx in range(collected_spans.cursor):
                if _do_spans_overlap(spans, span_idx,
                                     &collected_spans, coll_span_idx):
                    coll_width = abs(collected_spans.end[coll_span_idx] - collected_spans.beg[coll_span_idx])
                    if new_width < coll_width:
                        # Replace
                        collected_spans.terms[coll_span_idx] = spans[0].terms[span_idx]
                        collected_spans.posns[coll_span_idx] = spans[0].posns[span_idx]
                        collected_spans.beg[coll_span_idx] = spans[0].beg[span_idx]
                        collected_spans.end[coll_span_idx] = spans[0].end[span_idx]
                        overlaps = True
                        break
            if not overlaps:
                collected_spans.terms[collected_spans.cursor] = spans[0].terms[span_idx]
                collected_spans.posns[collected_spans.cursor] = spans[0].posns[span_idx]
                collected_spans.beg[collected_spans.cursor] = spans[0].beg[span_idx]
                collected_spans.end[collected_spans.cursor] = spans[0].end[span_idx]
                collected_spans.cursor += 1
    return collected_spans


cdef _span_freqs(DTYPE_t[:] posns,      # Flattened all terms in one array
                 DTYPE_t[:] lengths,
                 double[:] phrase_freqs,
                 DTYPE_t slop,
                 DTYPE_t key_mask,
                 DTYPE_t header_mask,
                 DTYPE_t key_bits,
                 DTYPE_t lsb_bits):
    """Get unscored spans, within 64 bits."""

    cdef np.int64_t set_idx = 0
    cdef np.int64_t curr_posn = 0
    cdef DTYPE_t payload_mask = ~header_mask
    cdef DTYPE_t payload_msb_mask = header_mask & ~key_mask

    cdef ActiveSpans spans = _new_active_spans()

    cdef np.uint64_t[:] curr_idx = np.zeros(64, dtype=np.uint64)
    cdef np.uint64_t[:] sum_popcount = np.zeros(64, dtype=np.uint64)
    cdef np.uint64_t curr_term_mask = 0
    cdef np.uint64_t posn_mask = 0
    cdef np.uint64_t num_terms = len(lengths) - 1
    cdef np.uint64_t term_ord = 0
    cdef np.uint64_t term = 0
    cdef np.uint64_t curr_key = 0
    cdef np.uint64_t last_key = 0
    cdef np.uint64_t payload_base = 0
    cdef np.uint64_t max_span_width = num_terms + slop
    cdef ActiveSpans collected_spans
    last_set_idx = 0
    cdef bint full = False

    for i in range(num_terms):
        curr_idx[i] = lengths[i]

    while curr_idx[0] < lengths[1]:
        # Read each term up to the next  doc
        for term_ord in range(num_terms):
            curr_key = ((posns[curr_idx[term_ord]] & key_mask) >> (64 - key_bits))
            sum_popcount[term_ord] = 0
            payload_base = 0
            while curr_idx[term_ord] < lengths[term_ord+1]:
                last_key = curr_key
                term = posns[curr_idx[term_ord]]
                payload_base = ((term & payload_msb_mask) >> lsb_bits) * lsb_bits
                term &= payload_mask
                curr_term_mask = 0x1 << term_ord
                sum_popcount[term_ord] += __builtin_popcountll(posns[curr_idx[term_ord]] & payload_mask)

                # Consume every position into every possible span
                while term != 0:
                    set_idx = _consume_lsb(&term)
                    curr_posn = set_idx + payload_base
                    posn_mask = _posn_mask(curr_posn)

                    spans.terms[spans.cursor] = curr_term_mask
                    spans.posns[spans.cursor] = posn_mask
                    spans.beg[spans.cursor] = curr_posn
                    spans.end[spans.cursor] = curr_posn

                    # Update existing spans
                    end = spans.cursor
                    spans.cursor += 1
                    for span_idx in range(end):
                        # Continue active spans
                        num_terms_visited = _num_terms(&spans, span_idx)
                        num_posns_visited = _num_posns(&spans, span_idx)
                        if num_terms_visited < num_terms and num_posns_visited == num_terms:
                            continue
                        spans.terms[span_idx] |= curr_term_mask
                        num_terms_visited_now = _num_terms(&spans, span_idx)
                        # New term
                        if num_terms_visited_now > num_terms_visited:
                            # Add position for new unique term
                            spans.posns[span_idx] |= posn_mask

                            new_unique_posns = _num_posns(&spans, span_idx)
                            proposed_width = abs(curr_posn - spans.beg[span_idx])
                            if (num_posns_visited == new_unique_posns) or proposed_width > max_span_width:
                                # Clear curr_term_mask and cancel this position, we've seen it before
                                spans.terms[span_idx] &= ~curr_term_mask
                                continue
                            if spans.cursor < 512:
                                spans.terms[spans.cursor] = spans.terms[span_idx]
                                spans.posns[spans.cursor] = (spans.posns[span_idx] & ~posn_mask)
                                spans.beg[spans.cursor] = spans.beg[span_idx]
                                spans.end[spans.cursor] = spans.end[span_idx]
                                spans.cursor += 1
                                full = False
                            else:
                                full = True

                            spans.end[span_idx] = curr_posn
                            span_width = _span_width(&spans, span_idx)
                            if span_width > max_span_width:
                                continue

                    if spans.cursor >= 512:
                        break
                    last_set_idx = set_idx
                curr_idx[term_ord] += 1
                if curr_idx[term_ord] < lengths[term_ord+1]:
                    curr_key = (posns[curr_idx[term_ord]] & key_mask) >> (64 - key_bits)
                if spans.cursor >= 512:
                    spans = _compact_spans(&spans, max_span_width)
                    if spans.cursor >= 512:
                        # Give up
                        # Read until key change
                        for i in range(curr_idx[term_ord], lengths[term_ord+1]):
                            term = posns[i]
                            curr_key = (term & key_mask) >> (64 - key_bits)
                            if curr_key != last_key:
                                curr_idx[term_ord] = i
                                break
                if curr_key != last_key:
                    break

        if full:
            min_popcount = 0
            for term_ord in range(num_terms):
                if min_popcount == 0 or sum_popcount[term_ord] < min_popcount:
                    min_popcount = sum_popcount[term_ord]
            phrase_freqs[last_key] += min_popcount
        else:
            # All terms consumed for doc
            collected_spans = _collect_spans(&spans, num_terms, max_span_width)
            phrase_freqs[last_key] += collected_spans.cursor

        # Reset
        spans = _new_active_spans()
        full = False


def span_search(np.ndarray[DTYPE_t, ndim=1] posns,
                np.ndarray[DTYPE_t, ndim=1] lengths,
                np.ndarray[double, ndim=1] phrase_freqs,
                DTYPE_t slop,
                DTYPE_t key_mask,
                DTYPE_t header_mask,
                DTYPE_t key_bits,
                DTYPE_t lsb_bits):
    _span_freqs(posns, lengths, phrase_freqs, slop, key_mask, header_mask, key_bits, lsb_bits)
