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

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t

# cdef extern from "x86intrin.h":
#     int __builtin_popcountll(unsigned long long x)


cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)

ctypedef uint64_t DTYPE_t

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

class PostProcess(Enum):
    NONE = 0
    SHIFT_POPCOUNT = 1

# For some reason this as an inline is faster than
# just doing the operation, despite all the python
# interactions added
cdef inline DTYPE_t mskd(DTYPE_t value, DTYPE_t mask):
    return value & mask


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


cdef void _binary_search(DTYPE_t[:] array,
                         DTYPE_t target,
                         DTYPE_t mask,
                         np.intp_t* i,
                         np.intp_t len):
    cdef DTYPE_t value = array[i[0]]
    target &= mask

    # If already at correct location or beyond
    if target <= value & mask:
        return

    cdef np.intp_t i_right = len - 1  # is always GREATER OR EQUAL
    cdef np.intp_t i_left = i[0]  # is always LESS than value

    cdef DTYPE_t right = array[i_right]
    if right & mask < target:
        i[0] = i_right
        return # indicate target value too large

    while i_left + 1 < i_right:
        i[0] = (i_right + i_left) // 2
        value = array[i[0]]

        if target <= value & mask:
            i_right = i[0]
        else:
            i_left = i[0]

    i[0] = i_right

# Python wrapper for binary search
def binary_search(np.ndarray[DTYPE_t, ndim=1] array,
                  DTYPE_t target,
                  DTYPE_t mask=ALL_BITS,
                  start=0):
    cdef np.intp_t i = start
    cdef np.intp_t len = array.shape[0]
    _binary_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)

cdef void _galloping_search(DTYPE_t[:] array,
                            DTYPE_t target,
                            DTYPE_t mask,
                            np.intp_t* i,
                            np.intp_t len):
    cdef DTYPE_t value = array[i[0]] & mask 
    target &= mask

    # If already at correct location or beyond
    if target <= value:
        return

    cdef np.intp_t delta = 1
    cdef np.intp_t i_prev = i[0]

    while value < target:
        i_prev = i[0]
        i[0] += delta
        if len <= i[0]:
            # Gallop jump reached end of array.
            i[0] = len - 1
            value = array[i[0]] & mask
            break

        value = array[i[0]] & mask
        # Increase step size.
        delta *= 2

    cdef np.intp_t higher = i[0] + 1  # Convert pointer position to length.
    i[0] = i_prev  # This is the lower boundary and the active counter.

    _binary_search(array, target, mask, i, higher)


def galloping_search(np.ndarray[DTYPE_t, ndim=1] array,
                     DTYPE_t target,
                     DTYPE_t mask=ALL_BITS,
                     start=0):
    cdef np.intp_t i = start
    cdef np.intp_t len = array.shape[0]
    _galloping_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)


cdef _intersection(DTYPE_t[:] lhs,
                   DTYPE_t[:] rhs,
                   DTYPE_t mask=ALL_BITS,
                   DTYPE_t post_process=0):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef np.intp_t i_result = 0
    cdef DTYPE_t value_prev = -1
    cdef DTYPE_t value_lhs = 0
    cdef DTYPE_t value_rhs = 0

    # Outputs as numpy arrays
    cdef np.uint64_t[:] results = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)
    cdef np.int64_t result_idx = 0
    cdef np.uint64_t[:] lhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)

    while i_lhs < len_lhs and i_rhs < len_rhs:
        # Use gallping search to find the first element in the right array
        value_lhs = lhs[i_lhs] & mask
        value_rhs = rhs[i_rhs] & mask

        # Advance LHS to RHS
        if value_lhs < value_rhs:
            if i_lhs >= len_lhs - 1:
                break
            i_result = i_lhs
            _galloping_search(lhs, value_rhs, mask, &i_result, len_lhs)
            value_lhs = lhs[i_result] & mask
            i_lhs = i_result
        # Advance RHS to LHS
        elif value_rhs < value_lhs:
            if i_rhs >= len_rhs - 1:
                break
            i_result = i_rhs
            _galloping_search(rhs, value_lhs, mask, &i_result, len_rhs)
            value_rhs = rhs[i_result] & mask
            i_rhs = i_result

        if value_lhs == value_rhs:
            if value_prev != value_lhs:
                # Not a dup so store it.
                results[result_idx] = value_lhs
                lhs_indices[result_idx] = i_lhs
                rhs_indices[result_idx] = i_rhs
                result_idx += 1
            value_prev = value_lhs
            i_lhs += 1
            i_rhs += 1

    # Get view of each result and return
    return np.asarray(results), np.asarray(lhs_indices), np.asarray(rhs_indices), result_idx


def _u64(lst) -> np.ndarray:
    return np.array(lst, dtype=np.uint64)


def intersect(np.ndarray[DTYPE_t, ndim=1] lhs,
              np.ndarray[DTYPE_t, ndim=1] rhs,
              DTYPE_t mask=ALL_BITS,
              post_process=PostProcess.NONE):
    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    result, indices_lhs, indices_rhs, result_idx = _intersection(lhs, rhs, mask, post_process.value)
    return result[:result_idx], indices_lhs[:result_idx], indices_rhs[:result_idx]


cdef _adjacent(DTYPE_t[:] lhs,
               DTYPE_t[:] rhs,
               DTYPE_t mask=ALL_BITS,
               DTYPE_t delta=1):
    # Find all LHS / RHS indices where LHS is 1 before RHS
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef np.intp_t i_result = 0
    cdef DTYPE_t value_prev = -1
    cdef DTYPE_t value_lhs = 0
    cdef DTYPE_t value_rhs = 0

    # Outputs as numpy arrays
    cdef np.int64_t result_idx = 0
    cdef np.uint64_t[:] lhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)

    # Read rhs until > delta
    while i_rhs < len_rhs and rhs[i_rhs] & mask == 0:
        i_rhs += 1

    while i_lhs < len_lhs and i_rhs < len_rhs:
        # Use gallping search to find the first element in the right array
        value_lhs = lhs[i_lhs] & mask
        value_rhs = rhs[i_rhs] & mask
        
        # Advance LHS to RHS
        if value_lhs < value_rhs - delta:
            if i_lhs >= len_lhs - 1:
                break
            i_result = i_lhs
            # lhs   0_  2*  2   lhs / rhs are at _, now advance to *
            # rhs   0   3_  3
            # Advance lhs to the 
            _galloping_search(lhs, value_rhs - delta, mask, &i_result, len_lhs)
            value_lhs = lhs[i_result] & mask
            i_lhs = i_result
        # Advance RHS to LHS
        elif value_rhs - delta < value_lhs:
            if i_rhs >= len_rhs - 1:
                break
            i_result = i_rhs
            # lhs   0    2_   2   lhs / rhs are at _, now advance to *
            # rhs   0_   3*   3    so that rhs is one past lhs
            _galloping_search(rhs, value_lhs + delta,
                              mask, &i_result, len_rhs)
            value_rhs = rhs[i_result] & mask
            i_rhs = i_result

        if value_lhs == value_rhs - delta:
            if value_prev != value_lhs:
                # Not a dup so store it.
                lhs_indices[result_idx] = i_lhs
                rhs_indices[result_idx] = i_rhs
                result_idx += 1
            value_prev = value_lhs
            i_lhs += 1
            i_rhs += 1

    # Get view of each result and return
    return np.asarray(lhs_indices), np.asarray(rhs_indices), result_idx


def adjacent(np.ndarray[DTYPE_t, ndim=1] lhs,
             np.ndarray[DTYPE_t, ndim=1] rhs,
             DTYPE_t mask=ALL_BITS):
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    if mask is None:
        mask = ALL_BITS
        delta = 1
    else:
        delta = mask & -mask  # lest significant set bit on mask

    indices_lhs, indices_rhs, result_idx = _adjacent(lhs, rhs, mask, delta)
    return indices_lhs[:result_idx], indices_rhs[:result_idx]


cdef _scan_unique(DTYPE_t[:] arr,
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


cdef _scan_unique_shifted(DTYPE_t[:] arr,
                          DTYPE_t arr_len,
                          DTYPE_t rshift):
    cdef DTYPE_t i = 0

    cdef np.uint64_t[:] result = np.empty(arr_len, dtype=np.uint64)
    cdef DTYPE_t result_idx = 0
    cdef DTYPE_t target = arr[i] >> rshift

    while i < arr_len:
        target = arr[i] >> rshift
        result[result_idx] = target
        result_idx += 1
        i += 1
        while i < arr_len and (arr[i] >> rshift) == target:
            i += 1

    return result, result_idx




def unique(np.ndarray[DTYPE_t, ndim=1] arr,
           DTYPE_t rshift=0):
    if rshift > 0:
        result, result_idx = _scan_unique_shifted(arr, arr.shape[0], rshift)
    else:
        result, result_idx = _scan_unique(arr, arr.shape[0])
    return np.array(result[:result_idx])
