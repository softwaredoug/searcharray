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

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF
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
        return  # indicate target value too large

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
