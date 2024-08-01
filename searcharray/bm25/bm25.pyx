# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
cimport numpy as np
import numpy as np


cdef void _bm25_score(float* term_freqs,
                      float* adj_doc_lens,
                      double idf,
                      long length):
    """Modify termfreqs in place changing to BM25 score."""
    for _ in range(length):
        term_freqs[0] /= (term_freqs[0] + adj_doc_lens[0])
        term_freqs[0] *= idf
        term_freqs += 1
        adj_doc_lens += 1


def bm25_score(np.ndarray[np.float32_t, ndim=1] term_freqs,
               np.ndarray[np.float32_t, ndim=1] adj_doc_lens,
               idf):
    cdef long length = term_freqs.shape[0]
    _bm25_score(&term_freqs[0],
                &adj_doc_lens[0],
                idf,
                length)
