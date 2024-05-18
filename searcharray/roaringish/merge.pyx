# These are modified versions of sortednp:
#   https://gitlab.sauerburger.com/frank/sortednp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
cimport numpy as np
import numpy as np


# cimport snp_ops
# from snp_ops cimport _galloping_search, DTYPE_t, ALL_BITS
cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport DTYPE_t, int64_t


cdef _merge_naive(DTYPE_t[:] lhs,
                  DTYPE_t[:] rhs):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef DTYPE_t value_lhs = 0
    cdef DTYPE_t value_rhs = 0

    # Outputs as numpy arrays
    cdef np.uint64_t[:] results = np.empty(len_lhs + len_rhs, dtype=np.uint64)
    cdef np.int64_t result_idx = 0

    while i_lhs < len_lhs and i_rhs < len_rhs:
        # Use gallping search to find the first element in the right array
        value_lhs = lhs[i_lhs]
        value_rhs = rhs[i_rhs]

        if value_lhs < value_rhs:
            results[result_idx] = value_lhs
            i_lhs += 1
        elif value_rhs < value_lhs:
            results[result_idx] = value_rhs
            i_rhs += 1
        else:
            results[result_idx] = value_lhs
            result_idx += 1
            results[result_idx] = value_rhs
            i_lhs += 1
            i_rhs += 1
        result_idx += 1

    return np.asarray(results), result_idx


cdef DTYPE_t _merge(DTYPE_t* lhs,
                    DTYPE_t* rhs,
                    DTYPE_t len_lhs,
                    DTYPE_t len_rhs,
                    DTYPE_t* results) nogil:
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[len_lhs]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_rhs_ptr = &rhs[len_rhs]
    cdef DTYPE_t* result_ptr = &results[0]

    # Copy elements from both arrays
    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        if lhs_ptr[0] < rhs_ptr[0]:
            result_ptr[0] = lhs_ptr[0]
            lhs_ptr += 1
        elif rhs_ptr[0] < lhs_ptr[0]:
            result_ptr[0] = rhs_ptr[0]
            rhs_ptr += 1
        else:
            result_ptr[0] = lhs_ptr[0]
            result_ptr += 1
            result_ptr[0] = rhs_ptr[0]
            lhs_ptr += 1
            rhs_ptr += 1
        result_ptr += 1

    # Copy remaining elements if one side not consumed
    while lhs_ptr == end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        result_ptr[0] = rhs_ptr[0]
        rhs_ptr += 1
        result_ptr += 1

    while rhs_ptr == end_rhs_ptr and lhs_ptr < end_lhs_ptr:
        result_ptr[0] = lhs_ptr[0]
        lhs_ptr += 1
        result_ptr += 1

    return result_ptr - &results[0]


cdef DTYPE_t _merge_w_drop(DTYPE_t* lhs,
                           DTYPE_t* rhs,
                           DTYPE_t len_lhs,
                           DTYPE_t len_rhs,
                           DTYPE_t* results) nogil:
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[len_lhs]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_rhs_ptr = &rhs[len_rhs]

    cdef DTYPE_t* result_ptr = &results[0]

    # Copy elements from both arrays
    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        if lhs_ptr[0] < rhs_ptr[0]:
            result_ptr[0] = lhs_ptr[0]
            lhs_ptr += 1
        elif rhs_ptr[0] < lhs_ptr[0]:
            result_ptr[0] = rhs_ptr[0]
            rhs_ptr += 1
        else:
            result_ptr[0] = lhs_ptr[0]
            lhs_ptr += 1
            rhs_ptr += 1
        result_ptr += 1

    # Copy remaining elements if one side not consumed
    while lhs_ptr == end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        result_ptr[0] = rhs_ptr[0]
        rhs_ptr += 1
        result_ptr += 1

    while rhs_ptr == end_rhs_ptr and lhs_ptr < end_lhs_ptr:
        result_ptr[0] = lhs_ptr[0]
        lhs_ptr += 1
        result_ptr += 1

    return result_ptr - &results[0]


def merge(np.ndarray[DTYPE_t, ndim=1] lhs,
          np.ndarray[DTYPE_t, ndim=1] rhs,
          bint drop_duplicates=False):
    # Outputs as numpy arrays
    cdef DTYPE_t result_idx

    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t lhs_len = lhs.shape[0]
    cdef DTYPE_t rhs_len = rhs.shape[0]
    cdef DTYPE_t[:] results = np.empty(lhs.shape[0] + rhs.shape[0], dtype=np.uint64)
    cdef DTYPE_t* result_ptr = &results[0]

    with nogil:
        if drop_duplicates:
            result_idx = _merge_w_drop(lhs_ptr, rhs_ptr,
                                       lhs_len, rhs_len,
                                       result_ptr)
        else:
            result_idx = _merge(lhs_ptr, rhs_ptr,
                                lhs_len, rhs_len,
                                result_ptr)

    return np.array(results[:result_idx])
