import cython
from cython.operator cimport dereference

from cpython.mem cimport PyMem_Calloc, PyMem_Realloc, PyMem_Free
from time import perf_counter

from libc.string cimport memcpy

import numpy as np
cimport numpy as np
from numpy cimport ndarray as np_ndarray

from searcharray.utils.mat_set import SparseMatSetBuilder
from searcharray.term_dict import TermDict

# Typedef uint64
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef unsigned int uint32_t
    ctypedef long long int64_t

cdef extern from "numpy/arrayobject.h":
    np_ndarray PyArray_SimpleNewFromData(int nd, np.npy_intp* dims, int typenum, void* data)

# Constants for term id ,doc id, term posn idxs
cdef uint64_t TERM_ID_IDX = 0
cdef uint64_t DOC_ID_IDX = 1
cdef uint64_t TERM_POSN_IDX = 2
cdef uint64_t TERM_OCCUR_IDX = 3

# Constant for initial terms per doc to allocate
cdef uint64_t INITIAL_TERMS_PER_DOC = 1000


cdef uint64_t* realloc64(uint64_t* ptr, uint64_t old_size):
    cdef uint64_t* new_ptr = <uint64_t*>PyMem_Calloc(old_size * 2, sizeof(uint64_t))
    memcpy(new_ptr, ptr, old_size * sizeof(uint64_t))
    PyMem_Free(ptr)
    return new_ptr


cdef uint32_t* realloc32(uint32_t* ptr, uint64_t old_size):
    cdef uint32_t* new_ptr = <uint32_t*>PyMem_Calloc(old_size * 2, sizeof(uint32_t))
    memcpy(new_ptr, ptr, old_size * sizeof(uint32_t))
    PyMem_Free(ptr)
    return new_ptr


cdef _tokenize(array, tokenizer):
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()

    cdef np.ndarray[np.uint64_t, ndim=1] doc_lens = np.zeros(len(array), dtype=np.uint64)

    cdef uint64_t idx = 0
    cdef uint64_t term_posn = 0
    cdef uint64_t term_id = 0
    cdef uint64_t doc_id = 0
    cdef uint64_t terms_w_posns_idx

    cdef uint64_t term_freq_capac = 1000000
    cdef uint64_t terms_w_posns_capac = INITIAL_TERMS_PER_DOC * len(array) * 4
    cdef uint64_t* term_freqs= <uint64_t*>PyMem_Calloc(term_freq_capac, sizeof(uint64_t))
    cdef uint32_t* terms_w_posns = <uint32_t*>PyMem_Calloc(terms_w_posns_capac, sizeof(uint32_t))
    cdef list tokens

    start = perf_counter()

    for doc_id, doc in enumerate(array):

        term_posn = 0
        tokens = [term_dict.add_term(term) for term in tokenizer(doc)]
        if len(term_dict) >= term_freq_capac:
            term_freqs = realloc64(term_freqs, term_freq_capac)
            term_freq_capac = term_id * 2

        # Reallocate outside hot loop if needed
        if (idx + len(tokens)) * 4 >= terms_w_posns_capac:
            terms_w_posns = realloc32(terms_w_posns, terms_w_posns_capac)
            terms_w_posns_capac *= 2

        for term_posn in range(len(tokens)):
            term_id = tokens[term_posn]
            terms_w_posns_idx = idx * 4
            terms_w_posns[terms_w_posns_idx + TERM_ID_IDX] = term_id
            terms_w_posns[terms_w_posns_idx + DOC_ID_IDX] = doc_id
            terms_w_posns[terms_w_posns_idx + TERM_POSN_IDX] = term_posn
            terms_w_posns[terms_w_posns_idx + TERM_OCCUR_IDX] = term_freqs[term_id]
            term_freqs[term_id] += 1

            idx += 1
    
        term_doc.append(np.unique(tokens))
        doc_lens[doc_id] = len(tokens)

    # Receive into numpy array
    cdef np.npy_intp dims[1]
    dims[0] = idx * 4
    cdef np_ndarray all_terms = PyArray_SimpleNewFromData(1, dims, np.NPY_UINT32, terms_w_posns)
    all_terms = all_terms.reshape((4, idx))
    
    print(f"Tokenized in {perf_counter() - start} seconds")
    # Check sum of term freqs is same as total number of terms
    # accum_sum = 0
    # for i in range(0, len(term_dict)):
    #     accum_sum += term_freqs[i]
    # assert accum_sum == idx, f"accum_sum {accum_sum} != idx {idx}"


    # Stable sort by term id
    # sort_indices = np.argsort(all_terms[0], kind='mergesort')
    # all_terms = all_terms[:, sort_indices]

    # Change term freqs to a cumulative sum
    for i in range(1, len(term_dict)):
        term_freqs[i] += term_freqs[i - 1]

    print(f"Cumsum in {perf_counter() - start} seconds")

    sort_indices = np.zeros(len(all_terms[0]), dtype=np.uint64)

    # Now we can reconstruct the correct, inverted sort order using the term freqs
    cdef uint64_t start_idx = 0
    actual_len = idx
    for idx in range(0, actual_len):
        term_id = all_terms[TERM_ID_IDX, idx]
        start_idx = term_freqs[term_id - 1] if term_id > 0 else 0
        dest = start_idx + all_terms[TERM_OCCUR_IDX, idx]
        # Swap into the correct position
        sort_indices[idx] = dest
        assert dest < actual_len, f"dest {dest} >= actual_len {actual_len} at {idx} - start:{start_idx} {all_terms[:, idx]}"
        # all_terms[:, [dest, src]] = all_terms[:, [src, dest]]
    
    print(f"Sort indices in {perf_counter() - start} seconds")
    
    # Slice off the unused portion of the array, drop occurences
    all_terms = all_terms[:3, :actual_len]

    # Stable sort by term id
    # sort_indices = np.argsort(all_terms[0], kind='mergesort')
    all_terms = all_terms[:, sort_indices]
    print(f"Slice in {perf_counter() - start} seconds")

    PyMem_Free(term_freqs)

    return all_terms, term_dict, term_doc, doc_lens


def tokenize(array, tokenizer):
    return _tokenize(array, tokenizer)
