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


cdef DTYPE_t _popcount_reduce_at(DTYPE_t[:] ids, DTYPE_t[:] payload,
                                 DTYPE_t[:] merged_ids, float[:] merged_counts):
    cdef DTYPE_t last_id = ids[0]
    cdef DTYPE_t popcount_sum = 0
    cdef DTYPE_t* merged_ids_ptr = &merged_ids[0]
    cdef DTYPE_t* payload_ptr = &payload[0]
    cdef DTYPE_t* ids_ptr = &ids[0]
    cdef float* merged_counts_ptr = &merged_counts[0]

    # We already have 0, now add new values
    while ids_ptr < &ids[0] + ids.shape[0]:
        if ids_ptr[0] != last_id:
            merged_ids_ptr[0] = last_id
            merged_counts_ptr[0] = popcount_sum
            popcount_sum = 0
            merged_ids_ptr += 1
            merged_counts_ptr += 1
        popcount_sum += __builtin_popcountll(payload_ptr[0])
        last_id = ids_ptr[0]
        payload_ptr += 1
        ids_ptr += 1
    # Save final value
    merged_ids_ptr[0] = last_id
    merged_counts_ptr[0] = popcount_sum
    return merged_ids_ptr - &merged_ids[0] + 1


def popcount_reduce_at(np.ndarray[DTYPE_t, ndim=1] ids,
                       np.ndarray[DTYPE_t, ndim=1] payload):
    """Write the sum of popcount of the payload at the indices in ids to the output array.

    Add one to each plus_one ids

    """
    if len(ids) != len(payload):
        raise ValueError("ids and payload must have the same length")
    if len(ids) == 0:
        return np.array([]), np.array([])
    merged_ids = np.empty(ids.shape[0], dtype=np.uint64)
    merged_counts = np.empty(ids.shape[0], dtype=np.float32)
    merged_len = _popcount_reduce_at(ids, payload, merged_ids, merged_counts)
    return np.array(merged_ids[:merged_len]), np.array(merged_counts[:merged_len])


cdef _key_sum_over(DTYPE_t[:] ids, DTYPE_t[:] count,
                   DTYPE_t[:] merged_ids, float[:] merged_counts):
    cdef DTYPE_t* ids_ptr = &ids[0]
    cdef DTYPE_t* count_ptr = &count[0]
    cdef DTYPE_t* merged_ids_ptr = &merged_ids[0]
    cdef float* merged_counts_ptr = &merged_counts[0]
    cdef DTYPE_t last_id = ids[0]
    cdef DTYPE_t payload_sum = 0

    while ids_ptr < &ids[0] + ids.shape[0]:
        if ids_ptr[0] != last_id:
            merged_ids_ptr[0] = last_id
            merged_counts_ptr[0] = payload_sum
            payload_sum = 0
            merged_ids_ptr += 1
            merged_counts_ptr += 1
        payload_sum += count_ptr[0]
        last_id = ids_ptr[0]
        ids_ptr += 1
        count_ptr += 1
    # Save final value
    merged_ids_ptr[0] = last_id
    merged_counts_ptr[0] = payload_sum
    return merged_ids_ptr - &merged_ids[0] + 1


def key_sum_over(np.ndarray[DTYPE_t, ndim=1] ids,
                 np.ndarray[DTYPE_t, ndim=1] count):
    """Write the last value of the payload at the indices in ids to the output array."""
    if len(ids) != len(count):
        raise ValueError("ids and count must have the same length")
    if len(ids) == 0:
        return np.array([]), np.array([])
    merged_ids = np.empty(ids.shape[0], dtype=np.uint64)
    merged_counts = np.empty(ids.shape[0], dtype=np.float32)
    merged_len = _key_sum_over(ids, count, merged_ids, merged_counts)
    return np.array(merged_ids[:merged_len]), np.array(merged_counts[:merged_len])


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
    cdef float* popcounts_ptr = &popcounts[0]
    cdef DTYPE_t* keys_ptr = &keys[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    cdef DTYPE_t last_key = arr_ptr[0] >> key_shift
    cdef DTYPE_t key = last_key
    keys_ptr[0] = last_key

    for _ in range(arr.shape[0]):
        key = arr_ptr[0] >> key_shift
        if key == last_key:
            popcounts_ptr[0] += __builtin_popcountll(arr_ptr[0] & value_mask)
        else:
            last_key = key
            popcounts_ptr += 1
            keys_ptr += 1
            # Init next key
            keys_ptr[0] = last_key
            popcounts_ptr[0] = __builtin_popcountll(arr_ptr[0] & value_mask)
        arr_ptr += 1
    return keys, popcounts, (keys_ptr - &keys[0] + 1)


# Branchless version using lookup table instead of if...
# Popcount reduce key-value pair
# for words 0xKKKKKKKK...KKKKVVVV...VVVV
# Returning two parallel arrays:
#   - keys   - the keys
#   - values - the popcount of the values with those keys
cdef _popcount64_reduce_nobranch(DTYPE_t[:] arr,
                                 DTYPE_t key_shift,
                                 DTYPE_t value_mask):
    cdef float[:] popcounts = np.zeros(arr.shape[0], dtype=np.float32)
    cdef DTYPE_t[:] keys = np.empty(arr.shape[0], dtype=np.uint64)
    cdef float* popcounts_ptr = &popcounts[0]
    cdef DTYPE_t* keys_ptr = &keys[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    cdef DTYPE_t last_key = arr_ptr[0] >> key_shift
    cdef DTYPE_t key = last_key

    keys_ptr[0] = last_key

    for _ in range(arr.shape[0]):
        key = arr_ptr[0] >> key_shift
        popcounts_ptr += (key != last_key)
        keys_ptr += (key != last_key)
        popcounts_ptr[0] += __builtin_popcountll(arr_ptr[0] & value_mask)
        keys_ptr[0] = key
        last_key = key
        arr_ptr += 1
    return keys, popcounts, (keys_ptr - &keys[0] + 1)


def popcount64_reduce(arr,
                      key_shift,
                      value_mask):
    cdef DTYPE_t[:] arr_view = arr
    if len(arr_view) == 0:
        return np.array([]), np.array([])
    keys, popcounts, results_idx = _popcount64_reduce(arr_view, key_shift, value_mask)
    return np.array(keys[:results_idx]), np.array(popcounts[:results_idx])
