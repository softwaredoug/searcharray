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


cdef _binary_search(np.ndarray[DTYPE_t, ndim=1] array,
                    DTYPE_t target,
                    DTYPE_t mask,
                    np.intp_t* i, np.intp_t len):
    cdef DTYPE_t value = array[i[0]]

    # If already at correct location or beyond
    if mskd(target, mask) <= mskd(value, mask):
        return

    cdef np.intp_t i_right = len - 1  # is always GREATER OR EQUAL
    cdef np.intp_t i_left = i[0]  # is always LESS than value

    cdef DTYPE_t right = array[i_right]
    if mskd(right, mask) < mskd(target, mask):
        i[0] = i_right
        return # indicate target value too large

    while i_left + 1 < i_right:
        i[0] = (i_right + i_left) // 2
        value = array[i[0]]

        if mskd(target, mask) <= mskd(value, mask):
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

cdef _galloping_search(np.ndarray[DTYPE_t, ndim=1] array,
                       DTYPE_t target,
                       DTYPE_t mask,
                       np.intp_t* i, np.intp_t len):
    cdef DTYPE_t value = array[i[0]]

    # If already at correct location or beyond
    if mskd(target, mask) <= mskd(value, mask):
        return False

    cdef np.intp_t delta = 1
    cdef np.intp_t i_prev = i[0]

    while mskd(value, mask) < mskd(target, mask):
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

    return _binary_search(array, target, mask, i, higher)


def galloping_search(np.ndarray[DTYPE_t, ndim=1] array,
                     DTYPE_t target,
                     DTYPE_t mask=ALL_BITS):
    cdef np.intp_t i = 0
    cdef np.intp_t len = array.shape[0]
    _galloping_search(array, target, mask, &i, len)
    return i, (array[i] & mask) == (target & mask)


cdef _intersection(np.ndarray[DTYPE_t, ndim=1] lhs,
                   np.ndarray[DTYPE_t, ndim=1] rhs,
                   DTYPE_t mask=ALL_BITS):
    cdef np.intp_t len_lhs = lhs.shape[0]
    cdef np.intp_t len_rhs = rhs.shape[0]
    cdef np.intp_t i_lhs = 0
    cdef np.intp_t i_rhs = 0
    cdef np.intp_t i_result = 0
    cdef DTYPE_t value_prev_masked = -1
    cdef DTYPE_t value_lhs_masked = 0
    cdef DTYPE_t value_rhs_masked = 0

    # Resulting lhs indices, rhs indices, and result indices as python lists:
    cdef list indices_lhs = []
    cdef list indices_rhs = []
    cdef list results = []

    print("Start intersection")

    while i_lhs < len_lhs and i_rhs < len_rhs:
        # Use gallping search to find the first element in the right array
        value_lhs = lhs[i_lhs]
        value_rhs = rhs[i_rhs]
        value_lhs_masked = mskd(value_lhs, mask)
        value_rhs_masked = mskd(value_rhs, mask)

        # Advance LHS to RHS
        if value_lhs_masked < value_rhs_masked:
            _galloping_search(lhs, value_rhs, mask, &i_result, len_lhs)
            print(f"Search lhs:{lhs} for {value_rhs} -- Found {i_result}")
            value_lhs = lhs[i_result]
            i_lhs = i_result
            value_lhs_masked = mskd(value_lhs, mask)
            # if value_lhs_masked != value_rhs_masked:
            #    print("EXIT-1")
            #     break
        # Advance RHS to LHS
        elif value_lhs_masked > value_rhs_masked:
            _galloping_search(rhs, value_lhs, rhs, &i_result, len_rhs)
            print(f"Search rhs:{rhs} for {value_lhs} -- Found {i_result}")
            value_rhs = rhs[i_result]
            i_rhs = i_result
            value_rhs_masked = mskd(value_rhs, mask)
            # if value_lhs_masked != value_rhs_masked:
            #     print("EXIT-2")
            #     break

        if value_lhs_masked == value_rhs_masked:
            print("Found intersection")
            if value_prev_masked != value_lhs_masked:
                # Not a dup so store it.
                results.append(value_lhs_masked)
                indices_lhs.append(i_lhs)
                indices_rhs.append(i_rhs)
        value_prev_masked = value_lhs_masked
        i_lhs += 1
        i_rhs += 1
    return results, indices_lhs, indices_rhs


def _u64(lst) -> np.ndarray:
    return np.array(lst, dtype=np.uint64)


def intersect(np.ndarray[DTYPE_t, ndim=1] lhs,
              np.ndarray[DTYPE_t, ndim=1] rhs,
              DTYPE_t mask=ALL_BITS):
    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    result, indices_lhs, indices_rhs = _intersection(lhs, rhs, mask)
    return _u64(result), _u64(indices_lhs), _u64(indices_rhs)
