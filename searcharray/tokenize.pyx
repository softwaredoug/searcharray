import cython

import numpy as np
cimport numpy as np

from searcharray.term_dict import TermDict
from searcharray.utils.mat_set import SparseMatSetBuilder

# Typedef uint64
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef long long int64_t

# Constants for term id ,doc id, term posn idxs
cdef uint64_t TERM_ID_IDX = 0
cdef uint64_t DOC_ID_IDX = 1
cdef uint64_t TERM_POSN_IDX = 2

# Constant for initial terms per doc to allocate
cdef uint64_t INITIAL_TERMS_PER_DOC = 100


cdef _tokenize(array, tokenizer):
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()

    cdef np.ndarray[np.uint64_t, ndim=2] all_terms = np.zeros((3, len(array) * INITIAL_TERMS_PER_DOC), dtype=np.uint64)

    # cdef list cols = []

    cdef uint64_t[:,:] terms_w_posns = all_terms

    cdef uint64_t doc_id = 0
    cdef uint64_t idx = 0
    cdef uint64_t term_posn = 0
    cdef uint64_t term_id = 0

    for doc_id, doc in enumerate(array):

        term_posn = 0
        cols = []
    
        for term_posn, term in enumerate(tokenizer(doc)):
            term_id = term_dict.add_term(term)
            terms_w_posns[TERM_ID_IDX, idx] = term_id
            terms_w_posns[DOC_ID_IDX, idx] = doc_id
            terms_w_posns[TERM_POSN_IDX, idx] = term_posn

            cols.append(term_id)

            idx += 1
            if idx >= len(all_terms[0]):
                all_terms = np.concatenate((all_terms,
                                            np.zeros((3, len(all_terms[0])), dtype=np.uint64)),
                                           axis=1)

                terms_w_posns = all_terms

        term_doc.append(np.unique(cols))

    # Slice off the unused portion of the array
    all_terms = all_terms[:, :idx]

    return all_terms, term_dict, term_doc


def tokenize(array, tokenizer):
    return _tokenize(array, tokenizer)
