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
from searcharray.roaringish.snp_ops cimport DTYPE_t


cdef _scan_unique_naive(DTYPE_t[:] arr,
                        DTYPE_t arr_len):
    cdef DTYPE_t i = 0

    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef DTYPE_t result_idx = 0
    cdef DTYPE_t target = arr[i]

    while i < arr_len:
        target = arr[i]
        result[result_idx] = target
        result_idx += 1
        i += 1
        while i < arr_len and arr[i] == target:
            i += 1

    return result, result_idx

cdef _scan_unique(DTYPE_t[:] arr,
                  DTYPE_t arr_len):
    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t* arr_end = &arr[arr_len-1]
    cdef DTYPE_t* target_ptr = arr_ptr

    while arr_ptr <= arr_end:
        target_ptr = arr_ptr
        result_ptr[0] = arr_ptr[0]
        result_ptr += 1
        arr_ptr += 1
        while arr_ptr <= arr_end and arr_ptr[0] == target_ptr[0]:
            arr_ptr += 1

    return result, result_ptr - &result[0]


cdef _scan_unique_gallop(DTYPE_t[:] arr,
                         DTYPE_t arr_len):
    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t* last_arr_ptr = &arr[0]
    cdef DTYPE_t* arr_end = &arr[arr_len-1]
    cdef DTYPE_t* target_ptr = arr_ptr
    cdef DTYPE_t delta = 1

    while arr_ptr <= arr_end:
        target_ptr = arr_ptr
        result_ptr[0] = arr_ptr[0]
        result_ptr += 1
        arr_ptr += 1
        delta = 1
        last_arr_ptr = arr_ptr
        # Gallop to first element that is not equal
        while arr_ptr <= arr_end and arr_ptr[0] == target_ptr[0]:
            last_arr_ptr = arr_ptr
            arr_ptr += delta
            delta *= 2
        # Linearly search for the first element that is not equal
        arr_ptr = last_arr_ptr
        if arr_ptr <= arr_end:
            while arr_ptr <= arr_end and arr_ptr[0] == target_ptr[0]:
                arr_ptr += 1

    return result, result_ptr - &result[0]


cdef _scan_unique_shifted(DTYPE_t[:] arr,
                          DTYPE_t arr_len,
                          DTYPE_t rshift):
    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t* arr_end = &arr[arr_len-1]
    cdef DTYPE_t  target_shifted = arr_ptr[0] >> rshift

    while arr_ptr <= arr_end:
        target_shifted = arr_ptr[0] >> rshift
        result_ptr[0] = target_shifted
        result_ptr += 1
        arr_ptr += 1
        while arr_ptr <= arr_end and (arr_ptr[0] >> rshift) == target_shifted:
            arr_ptr += 1

    return result, result_ptr - &result[0]


cdef _scan_unique_shifted_gallop(DTYPE_t[:] arr,
                                 DTYPE_t arr_len,
                                 DTYPE_t rshift):
    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t* last_arr_ptr = &arr[0]
    cdef DTYPE_t* arr_end = &arr[arr_len-1]
    cdef DTYPE_t  target_shifted = arr_ptr[0] >> rshift
    cdef DTYPE_t delta = 1

    while arr_ptr <= arr_end:
        target_shifted = arr_ptr[0] >> rshift
        result_ptr[0] = arr_ptr[0]
        result_ptr += 1
        arr_ptr += 1
        delta = 1
        last_arr_ptr = arr_ptr
        # Gallop to first element that is not equal
        while arr_ptr <= arr_end and (arr_ptr[0] >> rshift == target_shifted):
            last_arr_ptr = arr_ptr
            arr_ptr += delta
            delta *= 2
        # Linearly search for the first element that is not equal
        arr_ptr = last_arr_ptr
        if arr_ptr <= arr_end:
            while arr_ptr <= arr_end and (arr_ptr[0] >> rshift == target_shifted):
                arr_ptr += 1

    return result, result_ptr - &result[0]


def unique(np.ndarray[DTYPE_t, ndim=1] arr,
           DTYPE_t rshift=0):
    if rshift > 0:
        result, result_idx = _scan_unique_shifted(arr, arr.shape[0], rshift)
    else:
        result, result_idx = _scan_unique(arr, arr.shape[0])
    return np.array(result[:result_idx])
