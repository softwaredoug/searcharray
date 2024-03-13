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

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t

ctypedef uint64_t DTYPE_t

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

# For some reason this as an inline is faster than
# just doing the operation, despite all the python
# interactions added
cdef inline DTYPE_t mskd(DTYPE_t value, DTYPE_t mask):
    return value & mask


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
                  DTYPE_t mask=ALL_BITS):
    cdef np.intp_t i = 0
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
                     DTYPE_t mask=ALL_BITS):
    cdef np.intp_t i = 0
    cdef np.intp_t len = array.shape[0]
    _galloping_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)


cdef _intersection(DTYPE_t[:] lhs,
                   DTYPE_t[:] rhs,
                   DTYPE_t mask=ALL_BITS):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef np.intp_t i_result = 0
    cdef DTYPE_t value_prev = -1

    # Outputs as numpy arrays
    cdef np.uint64_t[:] results = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)
    cdef np.int64_t result_idx = 0
    cdef np.uint64_t[:] lhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_indices = np.empty(min(len_lhs, len_rhs), dtype=np.uint64)

    # print("INTERSECTING:")
    # print(f"lhs: {lhs}")
    # print(f"rhs: {rhs}")

    while i_lhs < len_lhs and i_rhs < len_rhs:
        # Use gallping search to find the first element in the right array
        value_lhs = lhs[i_lhs] & mask
        value_rhs = rhs[i_rhs] & mask

        # print("=====================================")
        # print(f"i_lhs: {i_lhs}, i_rhs: {i_rhs}")
        # print(f"vals: {value_lhs}, {value_rhs}")

        # Advance LHS to RHS
        if value_lhs < value_rhs:
            # print(f"Advance lhs to rhs: {value_lhs} -> {value_rhs}")
            if i_lhs >= len_lhs - 1:
                break
            _galloping_search(lhs, value_rhs, mask, &i_result, len_lhs)
            value_lhs = lhs[i_result] & mask
            # print(f"search - i_result: {i_result}, value_lhs: {value_lhs}")
            i_lhs = i_result
            if value_lhs != value_rhs:
                break
        # Advance RHS to LHS
        elif value_rhs < value_lhs:
            if i_rhs >= len_rhs - 1:
                break
            # print(f"Advance rhs to lhs: {value_rhs} -> {value_lhs}")
            _galloping_search(rhs, value_lhs, mask, &i_result, len_rhs)
            value_rhs = rhs[i_result] & mask
            # print(f"search - i_result: {i_result}, value_rhs: {value_rhs}")
            i_rhs = i_result
            if value_lhs != value_rhs:
                break

        if value_lhs == value_rhs:
            if value_prev != value_lhs:
                # Not a dup so store it.
                # print(f"Store: {lhs[i_lhs]}")
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
              DTYPE_t mask=ALL_BITS):
    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    result, indices_lhs, indices_rhs, result_idx = _intersection(lhs, rhs, mask)
    return result[:result_idx], indices_lhs[:result_idx], indices_rhs[:result_idx]
    # return _u64(result), _u64(indices_lhs), _u64(indices_rhs)
