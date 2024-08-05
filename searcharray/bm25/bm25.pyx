# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
cimport numpy as np
import numpy as np


cdef void _bm25_score(float* term_freqs,
                      float* doc_lens,
                      float idf,
                      float avg_doc_lens,
                      float k1,
                      float b,
                      long length) nogil:
    """Modify termfreqs in place changing to BM25 score."""
    cdef float one_minus_b = 1 - b
    for _ in range(length):
        term_freqs[0] = (
            term_freqs[0] / (term_freqs[0] + (k1 * (one_minus_b + (b * (doc_lens[0] / avg_doc_lens)))))
        ) * idf
        term_freqs += 1
        doc_lens += 1


def bm25_score(np.ndarray[np.float32_t, ndim=1] term_freqs,
               np.ndarray[np.float32_t, ndim=1] doc_lens,
               float avg_doc_lens,
               float idf,
               float k1,
               float b):
    cdef long length = term_freqs.shape[0]
    _bm25_score(&term_freqs[0],
                &doc_lens[0],
                idf,
                avg_doc_lens,
                k1,
                b,
                length)
