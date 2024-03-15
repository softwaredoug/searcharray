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

# cimport snp_ops
# from snp_ops cimport _galloping_search, DTYPE_t, ALL_BITS
cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)


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


cdef _intersect_keep(DTYPE_t[:] lhs,
                     DTYPE_t[:] rhs,
                     DTYPE_t mask=ALL_BITS):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef np.intp_t i_result = 0
    cdef DTYPE_t value_lhs = 0
    cdef DTYPE_t value_rhs = 0
    cdef DTYPE_t target = 0

    # Outputs as numpy arrays
    cdef np.uint64_t[:] lhs_indices = np.empty(len_lhs, dtype=np.uint64)
    cdef np.uint64_t lhs_indices_idx = 0
    cdef np.uint64_t[:] rhs_indices = np.empty(len_rhs, dtype=np.uint64)
    cdef np.uint64_t rhs_indices_idx = 0


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

        if value_rhs == value_lhs:
            target = value_lhs & mask
            # Store all LHS indices equal to RHS
            while (lhs[i_lhs] & mask) == target and i_lhs < len_lhs:
                lhs_indices[lhs_indices_idx] = i_lhs
                lhs_indices_idx += 1
                i_lhs += 1
            # Store all RHS equal to LHS
            while (rhs[i_rhs] & mask) == target and i_rhs < len_rhs:
                rhs_indices[rhs_indices_idx] = i_rhs
                rhs_indices_idx += 1
                i_rhs += 1

    # Get view of each result and return
    return np.asarray(lhs_indices), np.asarray(rhs_indices), lhs_indices_idx, rhs_indices_idx



cdef _intersect_drop(DTYPE_t[:] lhs,
                     DTYPE_t[:] rhs,
                     DTYPE_t mask=ALL_BITS):
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
                lhs_indices[result_idx] = i_lhs
                rhs_indices[result_idx] = i_rhs
                result_idx += 1
            value_prev = value_lhs
            i_lhs += 1
            i_rhs += 1

    # Get view of each result and return
    return np.asarray(lhs_indices), np.asarray(rhs_indices), result_idx


def _u64(lst) -> np.ndarray:
    return np.array(lst, dtype=np.uint64)


def intersect(np.ndarray[DTYPE_t, ndim=1] lhs,
              np.ndarray[DTYPE_t, ndim=1] rhs,
              DTYPE_t mask=ALL_BITS,
              bint drop_duplicates=True):
    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    if drop_duplicates:
        indices_lhs, indices_rhs, result_idx = _intersect_drop(lhs, rhs, mask)
        return indices_lhs[:result_idx], indices_rhs[:result_idx]
    else:
        indices_lhs, indices_rhs, indices_lhs_idx, indices_rhs_idx = _intersect_keep(lhs, rhs, mask)
        return indices_lhs[:indices_lhs_idx], indices_rhs[:indices_rhs_idx]


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
