# These are modified versions of sortednp:
#   https://gitlab.sauerburger.com/frank/sortednp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
cimport numpy as np
import numpy as np

ctypedef np.uint64_t DTYPE_t
DTYPE = np.uint64

cdef ALL_BITS = 0xFFFFFFFFFFFFFFFF


cdef inline mskd(DTYPE_t value, DTYPE_t mask):
    return value & mask


cdef _binary_search(DTYPE_t target,
                    DTYPE_t mask,
                    np.ndarray[DTYPE_t, ndim=1] array,
                    np.intp_t* i, np.intp_t len):
    cdef DTYPE_t value = array[i[0]]

    # If already at correct location or beyond
    if target <= value:
        return

    cdef np.intp_t i_right = len - 1  # is always GREATER OR EQUAL
    cdef np.intp_t i_left = i[0]  # is always LESS than value

    cdef DTYPE_t right = array[i_right]
    if right < target:
        i[0] = i_right
        return # indicate target value too large

    while i_left + 1 < i_right:
        i[0] = (i_right + i_left) // 2
        value = array[i[0]]

        if target <= value:
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
    _binary_search(target, mask, array, &i, len)
    return i, array[i] == target

cdef _galloping_search(DTYPE_t target,
                       DTYPE_t mask,
                       np.ndarray[DTYPE_t, ndim=1] array,
                       np.intp_t* i, np.intp_t len):
    cdef DTYPE_t value = array[i[0]]

    # If already at correct location or beyond
    if target <= value:
        return False

    cdef np.intp_t delta = 1
    cdef np.intp_t i_prev = i[0]

    while value < target:
        i_prev = i[0]
        i[0] += delta
        if len <= i[0]:
            # Gallop jump reached end of array.
            i[0] = len - 1
            value = array[i[0]]
            break

        value = array[i[0]]
        # Increase step size.
        delta *= 2

    cdef np.intp_t higher = i[0] + 1  # Convert pointer position to length.
    i[0] = i_prev  # This is the lower boundary and the active counter.

    return _binary_search(target, mask, array, i, higher)


def galloping_search(np.ndarray[DTYPE_t, ndim=1] array,
                     DTYPE_t target,
                     DTYPE_t mask=ALL_BITS):
    cdef np.intp_t i = 0
    cdef np.intp_t len = array.shape[0]
    _galloping_search(target, mask, array, &i, len)
    return i, array[i] == target
