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
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t, int64_t

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)


# Include mach performance timer
cdef extern from "mach/mach_time.h":
    uint64_t mach_absolute_time()


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
                         np.intp_t* idx_out,
                         np.intp_t len):
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
    cdef np.intp_t i = start
    cdef np.intp_t len = array.shape[0]
    _binary_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)



cdef inline void _galloping_search(DTYPE_t[:] array,
                                   DTYPE_t target,
                                   DTYPE_t mask,
                                   np.intp_t* idx_out,
                                   np.intp_t len):
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
    cdef np.intp_t i = start
    cdef np.intp_t len = array.shape[0]
    _galloping_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)


cdef _gallop_intersect_drop(DTYPE_t[:] lhs,
                            DTYPE_t[:] rhs,
                            DTYPE_t mask=ALL_BITS):
    """Two pointer approach to find the intersection of two sorted arrays."""
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[lhs.shape[0]]
    cdef DTYPE_t* end_rhs_ptr = &rhs[rhs.shape[0]]
    cdef DTYPE_t delta = 1
    cdef DTYPE_t last = -1
    cdef np.uint64_t[:] lhs_indices = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t* lhs_result_ptr = &lhs_indices[0]
    cdef np.uint64_t* rhs_result_ptr = &rhs_indices[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:

        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & mask) < (rhs_ptr[0] & mask):
            lhs_ptr+=delta
            delta <<= 1
        lhs_ptr -= (delta >> 1)
        delta = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & mask) < (lhs_ptr[0] & mask):
            rhs_ptr+=delta
            delta <<= 1
        rhs_ptr -= (delta >> 1)
        delta = 1

        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        if (lhs_ptr[0] & mask) < (rhs_ptr[0] & mask):
            lhs_ptr = lhs_ptr + 1
        elif (rhs_ptr[0] & mask) < (lhs_ptr[0] & mask):
            rhs_ptr = rhs_ptr + 1
        else:
            # If here values equal, collect
            if (last & mask) != (lhs_ptr[0] & mask):
                lhs_result_ptr[0] = lhs_ptr - &lhs[0]
                rhs_result_ptr[0] = rhs_ptr - &rhs[0]
                last = lhs_ptr[0]
                lhs_result_ptr += 1
                rhs_result_ptr += 1
            lhs_ptr += 1
            rhs_ptr += 1

    return np.asarray(lhs_indices), np.asarray(rhs_indices), lhs_result_ptr - &lhs_indices[0]


cdef _gallop_intersect_keep(DTYPE_t[:] lhs,
                            DTYPE_t[:] rhs,
                            DTYPE_t mask=ALL_BITS):
    """Two pointer approach to find the intersection of two sorted arrays."""
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[lhs.shape[0]]
    cdef DTYPE_t* end_rhs_ptr = &rhs[rhs.shape[0]]
    cdef DTYPE_t delta = 1
    cdef DTYPE_t target = -1
    cdef np.uint64_t[:] lhs_indices = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t* lhs_result_ptr = &lhs_indices[0]
    cdef np.uint64_t* rhs_result_ptr = &rhs_indices[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & mask) < (rhs_ptr[0] & mask):
            lhs_ptr += delta
            delta <<= 1
        lhs_ptr -= (delta >> 1)
        delta = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & mask) < (lhs_ptr[0] & mask):
            rhs_ptr += delta
            delta <<= 1
        rhs_ptr -= (delta >> 1)
        delta = 1

        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        if (lhs_ptr[0] & mask) < (rhs_ptr[0] & mask):
            lhs_ptr += 1
        elif (rhs_ptr[0] & mask) < (lhs_ptr[0] & mask):
            rhs_ptr += 1
        else:
            target = lhs_ptr[0] & mask
            # Store all LHS indices equal to RHS
            while (lhs_ptr[0] & mask) == target and lhs_ptr < end_lhs_ptr:
                lhs_result_ptr[0] = lhs_ptr - &lhs[0]; lhs_result_ptr += 1
                lhs_ptr += 1
            # Store all RHS equal to LHS
            while (rhs_ptr[0] & mask) == target and rhs_ptr < end_rhs_ptr:
                rhs_result_ptr[0] = rhs_ptr - &rhs[0]; rhs_result_ptr += 1
                rhs_ptr += 1

        # If delta 
        # Either we read past the array, or 

    return np.asarray(lhs_indices), np.asarray(rhs_indices), lhs_result_ptr - &lhs_indices[0], rhs_result_ptr - &rhs_indices[0]


cdef _gallop_adjacent(DTYPE_t[:] lhs,
                      DTYPE_t[:] rhs,
                      DTYPE_t mask=ALL_BITS,
                      DTYPE_t delta=1):
    # Find all LHS / RHS indices where LHS is 1 before RHS
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[lhs.shape[0]]
    cdef DTYPE_t* end_rhs_ptr = &rhs[rhs.shape[0]]
    cdef DTYPE_t lhs_delta = 1
    cdef DTYPE_t rhs_delta = 1
    cdef DTYPE_t last = -1
    cdef np.uint64_t[:] lhs_indices = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t* lhs_result_ptr = &lhs_indices[0]
    cdef np.uint64_t* rhs_result_ptr = &rhs_indices[0]
    
    # Read rhs until > delta
    while rhs_ptr < end_rhs_ptr and rhs_ptr[0] & mask == 0:
        rhs_ptr += 1

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        lhs_delta = 1
        rhs_delta = 1

        # Gallop, but instead check is:
        # if value_lhs < value_rhs - delta:
        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & mask) < ((rhs_ptr[0] & mask) - delta):
            lhs_ptr += lhs_delta
            lhs_delta <<= 1
        lhs_ptr -= (lhs_delta >> 1)
        while rhs_ptr < end_rhs_ptr and ((rhs_ptr[0] & mask) - delta) < (lhs_ptr[0] & mask):
            rhs_ptr += rhs_delta
            rhs_delta <<= 1
        rhs_ptr -= (rhs_delta >> 1)

        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        if (lhs_ptr[0] & mask) < ((rhs_ptr[0] & mask) - delta):
            lhs_ptr += 1
        elif ((rhs_ptr[0] & mask) - delta) < (lhs_ptr[0] & mask):
            rhs_ptr += 1
        else:
            if (lhs_ptr[0] & mask) == ((rhs_ptr[0] & mask) - delta):
                if (last & mask) != (lhs_ptr[0] & mask):
                    lhs_result_ptr[0] = lhs_ptr - &lhs[0]
                    rhs_result_ptr[0] = rhs_ptr - &rhs[0]
                    last = lhs_ptr[0]
                    lhs_result_ptr += 1
                    rhs_result_ptr += 1
                lhs_ptr += 1
                rhs_ptr += 1

        # If delta 
        # Either we read past the array, or 

    return np.asarray(lhs_indices), np.asarray(rhs_indices), lhs_result_ptr - &lhs_indices[0]



def _u64(lst) -> np.ndarray:
    return np.array(lst, dtype=np.uint64)


def save_input(lhs, rhs, mask):
    np.save(f"lhs_{len(lhs)}.npy", lhs)
    np.save(f"rhs_{len(lhs)}.npy", rhs)
    np.save(f"mask_{len(lhs)}.npy", mask)


def intersect(np.ndarray[DTYPE_t, ndim=1] lhs,
              np.ndarray[DTYPE_t, ndim=1] rhs,
              DTYPE_t mask=ALL_BITS,
              bint drop_duplicates=True):
    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    if drop_duplicates:
        # save_input(lhs, rhs, mask)
        indices_lhs, indices_rhs, result_idx = _gallop_intersect_drop(lhs, rhs, mask)
        return indices_lhs[:result_idx], indices_rhs[:result_idx]
    else:
        indices_lhs, indices_rhs, indices_lhs_idx, indices_rhs_idx = _gallop_intersect_keep(lhs, rhs, mask)
        return indices_lhs[:indices_lhs_idx], indices_rhs[:indices_rhs_idx]


def adjacent(np.ndarray[DTYPE_t, ndim=1] lhs,
             np.ndarray[DTYPE_t, ndim=1] rhs,
             DTYPE_t mask=ALL_BITS):
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    if mask is None:
        mask = ALL_BITS
        delta = 1
    else:
        delta = (mask & -mask)  # lest significant set bit on mask

    indices_lhs, indices_rhs, result_idx = _gallop_adjacent(lhs, rhs, mask, delta)
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
