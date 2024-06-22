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


cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)

    int __builtin_ctzll(unsigned long long x)
    int __builtin_clzll(unsigned long long x)


# Include mach performance timer
# cdef extern from "mach/mach_time.h":
#     uint64_t mach_absolute_time()


cdef popcount64_arr(DTYPE_t[:] arr):
    cdef np.uint64_t[:] result = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef DTYPE_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    for _ in range(arr.shape[0]):
        result_ptr[0] = __builtin_popcountll(arr_ptr[0])
        result_ptr += 1
        arr_ptr += 1
    return result


cdef ctz_arr(DTYPE_t[:] arr):
    cdef np.uint64_t[:] result = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef DTYPE_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    for _ in range(arr.shape[0]):
        result_ptr[0] = __builtin_ctzll(arr_ptr[0])
        result_ptr += 1
        arr_ptr += 1
    return result


cdef clz_arr(DTYPE_t[:] arr):
    cdef np.uint64_t[:] result = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef DTYPE_t* result_ptr = &result[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    for _ in range(arr.shape[0]):
        result_ptr[0] = __builtin_clzll(arr_ptr[0])
        result_ptr += 1
        arr_ptr += 1
    return result


cdef popcount64_arr_naive(DTYPE_t[:] arr):
    cdef np.uint64_t[:] result = np.empty(arr.shape[0], dtype=np.uint64)
    cdef int i = 0

    for i in range(arr.shape[0]):
        result[i] = __builtin_popcountll(arr[i])
    return result


def popcount64(np.ndarray[DTYPE_t, ndim=1] arr):
    """Count the number of set bits in a 64-bit integer."""
    return np.array(popcount64_arr(arr))


cdef _popcount_reduce_at(DTYPE_t[:] ids, DTYPE_t[:] payload, double[:] output):
    cdef DTYPE_t idx = 1
    cdef DTYPE_t popcount_sum = __builtin_popcountll(payload[0])

    # We already have 0, now add new values
    while idx < ids.shape[0]:
        if ids[idx] != ids[idx - 1]:
            output[ids[idx - 1]] = popcount_sum
            popcount_sum = 0
        popcount_sum += __builtin_popcountll(payload[idx])
        idx += 1
    # Save final value
    output[ids[idx - 1]] = popcount_sum


def popcount_reduce_at(np.ndarray[DTYPE_t, ndim=1] ids,
                       np.ndarray[DTYPE_t, ndim=1] payload,
                       np.ndarray[np.float64_t, ndim=1] output):
    """Write the sum of popcount of the payload at the indices in ids to the output array."""
    if len(ids) != len(payload):
        raise ValueError("ids and payload must have the same length")
    return np.array(_popcount_reduce_at(ids, payload, output))


cdef _key_sum_over(DTYPE_t[:] ids, DTYPE_t[:] count, double[:] output):
    cdef DTYPE_t i = 0
    for i in range(ids.shape[0]):
        output[ids[i]] += count[i]


def key_sum_over(np.ndarray[DTYPE_t, ndim=1] ids,
                 np.ndarray[DTYPE_t, ndim=1] count,
                 np.ndarray[np.float64_t, ndim=1] output):
    """Write the last value of the payload at the indices in ids to the output array."""
    _key_sum_over(ids, count, output)


# Popcount reduce key-value pair
# for words 0xKKKKKKKK...KKKKVVVV...VVVV
# Returning two parallel arrays:
#   - keys   - the keys
#   - values - the popcount of the values with those keys
cdef _popcount64_reduce(DTYPE_t[:] arr,
                        DTYPE_t key_shift,
                        DTYPE_t value_mask):
    cdef float[:] popcounts = np.zeros(arr.shape[0], dtype=np.float32)
    cdef DTYPE_t[:] keys = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef float* popcounts_ptr = &popcounts[0]
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


def popcount64_reduce(arr,
                      key_shift,
                      value_mask):
    cdef DTYPE_t[:] arr_view = arr
    keys, popcounts, results_idx = _popcount64_reduce(arr_view, key_shift, value_mask)
    return np.array(keys[:results_idx]), np.array(popcounts[:results_idx])
