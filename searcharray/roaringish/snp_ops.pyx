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
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t, int64_t

from libc.stdlib cimport malloc, free

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)


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


cdef popcount64_arr_naive(DTYPE_t[:] arr):
    cdef np.uint64_t[:] result = np.empty(arr.shape[0], dtype=np.uint64)
    cdef int i = 0

    for i in range(arr.shape[0]):
        result[i] = __builtin_popcountll(arr[i])
    return result


def popcount64(arr):
    return np.array(popcount64_arr(arr))


cdef _popcount_reduce_at(DTYPE_t[:] ids, DTYPE_t[:] payload, double[:] output):
    cdef DTYPE_t idx = 1
    cdef DTYPE_t popcount_sum = __builtin_popcountll(payload[0])
    cdef DTYPE_t result_idx = 1

    # We already have 0, now add new values
    while idx < ids.shape[0]:
        if ids[idx] != ids[idx - 1]:
            output[ids[idx - 1]] = popcount_sum
            popcount_sum = 0
        popcount_sum += __builtin_popcountll(payload[idx])
        idx += 1
    # Save final value
    output[ids[idx - 1]] = popcount_sum


def popcount_reduce_at(ids, payload, output):
    """Write the sum of popcount of the payload at the indices in ids to the output array."""
    if len(ids) != len(payload):
        raise ValueError("ids and payload must have the same length")
    return np.array(_popcount_reduce_at(ids, payload, output))



cdef void _binary_search(DTYPE_t[:] array,
                         DTYPE_t target,
                         DTYPE_t mask,
                         np.uint64_t* idx_out,
                         np.uint64_t len):
    """Write to idx_out the index of the first element in array that is
       greater or equal to target (masked)."""
    cdef DTYPE_t value = array[idx_out[0]]
    target &= mask

    # If already at correct location or beyond
    if target <= value & mask:
        return

    cdef np.intp_t i_right = len - 1  # is always GREATER OR EQUAL
    cdef np.intp_t i_left = idx_out[0]  # is always LESS than value

    if array[i_right] & mask < target:
        idx_out[0] = i_right
        return # indicate target value too large

    while i_left + 1 < i_right:
        idx_out[0] = (i_right + i_left) // 2  # midpoint
        value = array[idx_out[0]]

        if target <= value & mask:
            i_right = idx_out[0]
        else:
            i_left = idx_out[0]

    idx_out[0] = i_right

# Python wrapper for binary search
def binary_search(np.ndarray[DTYPE_t, ndim=1] array,
                  DTYPE_t target,
                  DTYPE_t mask=ALL_BITS,
                  start=0):
    cdef np.uint64_t i = start
    cdef np.uint64_t len = array.shape[0]
    _binary_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)



cdef inline void _galloping_search(DTYPE_t[:] array,
                                   DTYPE_t target,
                                   DTYPE_t mask,
                                   np.uint64_t* idx_out,
                                   np.uint64_t len):
    cdef DTYPE_t value = array[idx_out[0]] & mask 
    target &= mask

    # If already at correct location or beyond
    if target <= value:
        return

    cdef np.intp_t delta = 1
    cdef DTYPE_t end = len - 1
    cdef np.intp_t i_prev = idx_out[0]

    while value < target:
        i_prev = idx_out[0]
        idx_out[0] += delta
        if len <= idx_out[0]:
            # Gallop jump reached end of array.
            idx_out[0] = end
            value = array[idx_out[0]] & mask
            break

        value = array[idx_out[0]] & mask
        # Increase step size.
        delta *= 2

    cdef np.intp_t i_right = idx_out[0] + 1  # Convert pointer position to length.
    idx_out[0] = i_prev  # This is the lower boundary and the active counter.

    # i_prev ~ i_left to save a variable
    #
    # _binary_search(array, target, mask, i, higher)
    # Inline binary search without the checks
    # length is one past current posn
    # left is i_prev
    while i_prev + 1 < i_right :
        idx_out[0] = (i_right + i_prev) // 2  # midpoint
        value = array[idx_out[0]] & mask
        if target <= value:
            i_right = idx_out[0]
        else:
            i_prev = idx_out[0]
    idx_out[0] = i_right


# This is just a baseline function for measuring how
# fast we can expect a naive collection
cdef _count_odds(np.ndarray[DTYPE_t, ndim=1] lhs, np.ndarray[DTYPE_t, ndim=1] rhs):
    cdef int i = 0
    cdef int count = 0
    for i in range(rhs.shape[0]):
        if (rhs[i] & 1) or (i < lhs.shape[0] and (lhs[i] & 1)):
            count += 1
    return count


def count_odds(np.ndarray[DTYPE_t, ndim=1] lhs, np.ndarray[DTYPE_t, ndim=1] rhs):
    # Make sure lhs is smallest
    if lhs.shape[0] > rhs.shape[0]:
        lhs, rhs = rhs, lhs
    return _count_odds(lhs, rhs)


def galloping_search(np.ndarray[DTYPE_t, ndim=1] array,
                     DTYPE_t target,
                     DTYPE_t mask=ALL_BITS,
                     start=0):
    cdef np.uint64_t i = start
    cdef np.uint64_t len = array.shape[0]
    _galloping_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)

def _u64(lst) -> np.ndarray:
    return np.array(lst, dtype=np.uint64)


def save_input(lhs, rhs, mask):
    np.save(f"lhs_{len(lhs)}.npy", lhs)
    np.save(f"rhs_{len(lhs)}.npy", rhs)
    np.save(f"mask_{len(lhs)}.npy", mask)



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


cdef _merge(DTYPE_t[:] lhs,
            DTYPE_t[:] rhs):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]

    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[len_lhs]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_rhs_ptr = &rhs[len_rhs]

    # Outputs as numpy arrays
    cdef np.uint64_t[:] results = np.empty(len_lhs + len_rhs, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &results[0]

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

    return np.asarray(results), result_ptr - &results[0]


cdef _merge_w_drop(DTYPE_t[:] lhs,
                   DTYPE_t[:] rhs):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]

    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[len_lhs]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_rhs_ptr = &rhs[len_rhs]

    # Outputs as numpy arrays
    cdef np.uint64_t[:] results = np.empty(len_lhs + len_rhs, dtype=np.uint64)
    cdef np.uint64_t* result_ptr = &results[0]

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

    return np.asarray(results), result_ptr - &results[0]




def merge(np.ndarray[DTYPE_t, ndim=1] lhs,
          np.ndarray[DTYPE_t, ndim=1] rhs,
          bint drop_duplicates=False):
    if drop_duplicates:
        result, result_idx = _merge_w_drop(lhs, rhs)
    else:
        result, result_idx = _merge(lhs, rhs)
    return np.array(result[:result_idx])
