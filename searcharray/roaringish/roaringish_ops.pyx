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
from enum import Enum

cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t


cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF


cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)

# Popcount reduce key-value pair
# for words 0xKKKKKKKK...KKKKVVVV...VVVV
# Returning two parallel arrays:
#   - keys   - the keys
#   - values - the popcount of the values with those keys
cdef _popcount64_reduce(DTYPE_t[:] arr,
                        DTYPE_t key_shift,
                        DTYPE_t value_mask):
    cdef DTYPE_t[:] popcounts = np.zeros(arr.shape[0], dtype=np.uint64)
    cdef DTYPE_t[:] keys = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef DTYPE_t* popcounts_ptr = &popcounts[0]
    cdef DTYPE_t* keys_ptr = &keys[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t last_key = 0xFFFFFFFFFFFFFFFF

    for _ in range(arr.shape[0]):
        popcounts_ptr[0] += __builtin_popcountll(arr_ptr[0] & value_mask)
        if arr_ptr[0] >> key_shift != last_key:
            last_key = arr_ptr[0] >> key_shift
            keys_ptr[0] = last_key
            popcounts_ptr += 1
            keys_ptr += 1
        arr_ptr += 1
    return keys, popcounts, keys_ptr - &keys[0]


def popcount64_reduce(arr, key_shift, value_mask):
    cdef DTYPE_t[:] arr_view = arr
    keys, popcounts, results_idx = _popcount64_reduce(arr_view, key_shift, value_mask)
    return np.array(keys[:results_idx]), np.array(popcounts[:results_idx])
