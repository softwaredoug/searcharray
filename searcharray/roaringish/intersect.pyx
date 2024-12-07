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
cimport libc.stdint
from libc.stdio cimport printf
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
cimport cython


cdef extern from "timer.h":
    cdef uint64_t timestamp() nogil
    cdef void print_elapsed(uint64_t start, const char* msg) nogil


cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport DTYPE_t

cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF

# Struct with intersect args
cdef struct intersect_args:
    DTYPE_t* lhs
    DTYPE_t* rhs
    DTYPE_t lhs_stride
    DTYPE_t rhs_stride
    DTYPE_t lhs_len
    DTYPE_t rhs_len
    DTYPE_t* lhs_out
    DTYPE_t* rhs_out
    DTYPE_t mask

ctypedef intersect_args intersect_args_t

cdef DTYPE_t _gallop_intersect_drop(intersect_args_t args) nogil:
    """Two pointer approach to find the intersection of two sorted arrays."""
    cdef DTYPE_t* lhs_ptr = &args.lhs[0]
    cdef DTYPE_t* rhs_ptr = &args.rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &args.lhs[args.lhs_len]
    cdef DTYPE_t* end_rhs_ptr = &args.rhs[args.rhs_len]
    cdef DTYPE_t gallop = 1
    cdef DTYPE_t last = -1
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:

        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & args.mask) < (rhs_ptr[0] &args.mask):
            lhs_ptr+= (gallop * args.lhs_stride)
            gallop *= 2
        lhs_ptr -= ((gallop // 2) * args.lhs_stride)
        gallop = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr+= (gallop * args.rhs_stride)
            gallop *= 2
        rhs_ptr -= ((gallop // 2) * args.rhs_stride)
        gallop = 1

        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        if (lhs_ptr[0] & args.mask) < (rhs_ptr[0] & args.mask):
            lhs_ptr = lhs_ptr + args.lhs_stride
        elif (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr = rhs_ptr + args.rhs_stride
        else:
            # If here values equal, collect
            if (last & args.mask) != (lhs_ptr[0] & args.mask):
                lhs_result_ptr[0] = (lhs_ptr - &args.lhs[0]) / args.lhs_stride
                rhs_result_ptr[0] = (rhs_ptr - &args.rhs[0]) / args.rhs_stride
                last = lhs_ptr[0]
                lhs_result_ptr += 1
                rhs_result_ptr += 1
            lhs_ptr += args.lhs_stride
            rhs_ptr += args.rhs_stride

    return lhs_result_ptr - &args.lhs_out[0]


cdef void _gallop_intersect_keep(intersect_args_t args,
                                 DTYPE_t* lhs_out_len,
                                 DTYPE_t* rhs_out_len) noexcept nogil:
    """Two pointer approach to find the intersection of two sorted arrays."""
    cdef DTYPE_t* lhs_ptr = &args.lhs[0]
    cdef DTYPE_t* rhs_ptr = &args.rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &args.lhs[args.lhs_len]
    cdef DTYPE_t* end_rhs_ptr = &args.rhs[args.rhs_len]
    cdef DTYPE_t gallop = 1
    cdef DTYPE_t target = -1
    # cdef np.uint64_t[:] lhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    # cdef np.uint64_t[:] rhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & args.mask) < (rhs_ptr[0] & args.mask):
            lhs_ptr += (gallop * args.lhs_stride)
            gallop <<= 1
        lhs_ptr -= ((gallop >> 1) * args.lhs_stride)
        gallop = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr += (gallop * args.rhs_stride)
            gallop <<= 1
        rhs_ptr -= ((gallop >> 1) * args.rhs_stride)
        gallop = 1

        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        if (lhs_ptr[0] & args.mask) < (rhs_ptr[0] & args.mask):
            lhs_ptr += args.lhs_stride
        elif (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr += args.rhs_stride
        else:
            target = lhs_ptr[0] & args.mask
            # Store all LHS indices equal to RHS
            while (lhs_ptr[0] & args.mask) == target and lhs_ptr < end_lhs_ptr:
                lhs_result_ptr[0] = (lhs_ptr - &args.lhs[0]) / args.lhs_stride
                lhs_result_ptr += 1
                lhs_ptr += args.lhs_stride
            # Store all RHS equal to LHS
            while (rhs_ptr[0] & args.mask) == target and rhs_ptr < end_rhs_ptr:
                rhs_result_ptr[0] = (rhs_ptr - &args.rhs[0]) / args.rhs_stride
                rhs_result_ptr += 1
                rhs_ptr += args.rhs_stride

        # If delta
        # Either we read past the array, or

    lhs_out_len[0] = lhs_result_ptr - &args.lhs_out[0]
    rhs_out_len[0] = rhs_result_ptr - &args.rhs_out[0]


cdef DTYPE_t _gallop_adjacent(DTYPE_t* lhs,
                              DTYPE_t* rhs,
                              DTYPE_t lhs_len,
                              DTYPE_t rhs_len,
                              DTYPE_t* lhs_out,
                              DTYPE_t* rhs_out,
                              DTYPE_t mask=ALL_BITS,
                              DTYPE_t delta=1) noexcept nogil:
    # Find all LHS / RHS indices where LHS is 1 before RHS
    cdef DTYPE_t* lhs_ptr = &lhs[0]
    cdef DTYPE_t* rhs_ptr = &rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &lhs[lhs_len]
    cdef DTYPE_t* end_rhs_ptr = &rhs[rhs_len]
    cdef DTYPE_t lhs_gallop = 1
    cdef DTYPE_t rhs_gallop = 1
    cdef DTYPE_t last = -1
    cdef DTYPE_t* lhs_result_ptr = &lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &rhs_out[0]

    # Read rhs until > delta
    while rhs_ptr < end_rhs_ptr and rhs_ptr[0] & mask == 0:
        rhs_ptr += 1

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        lhs_gallop = 1
        rhs_gallop = 1

        # Gallop, but instead check is:
        # if value_lhs < value_rhs - delta:
        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & mask) < ((rhs_ptr[0] & mask) - delta):
            lhs_ptr += lhs_gallop
            lhs_gallop <<= 1
        lhs_ptr -= (lhs_gallop >> 1)
        while rhs_ptr < end_rhs_ptr and ((rhs_ptr[0] & mask) - delta) < (lhs_ptr[0] & mask):
            rhs_ptr += rhs_gallop
            rhs_gallop <<= 1
        rhs_ptr -= (rhs_gallop >> 1)

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

    return lhs_result_ptr - &lhs_out[0]


# Can you gallop adjacent and intersect at the same time?
#
# lhs      1     5       9
# rhs        2   5 6 7     10
#
# With a scan you would be able to
#
# lhs      1*      5
# rhs         2*   5   6
#          ----
# lhs      1       5*
# rhs         2*   5   6
#
# lhs      1       5*
# rhs         2    5*  6
#                  ---
# lhs      1       5*        ...   99 100        ... 139     150
# rhs         2    5   6*    ...      100        ...         150
#                      ---

cdef DTYPE_t _gallop_int_and_adj_drop(intersect_args_t args,
                                      DTYPE_t delta,
                                      DTYPE_t* adj_lhs_out,
                                      DTYPE_t* adj_rhs_out,
                                      DTYPE_t* adj_out_len) nogil:
    """Galloping approach to find the intersection w/ adjacents of two sorted arrays."""
    cdef DTYPE_t* lhs_ptr = &args.lhs[0]
    cdef DTYPE_t* rhs_ptr = &args.rhs[0]
    cdef DTYPE_t* end_lhs_ptr = &args.lhs[args.lhs_len]
    cdef DTYPE_t* end_rhs_ptr = &args.rhs[args.rhs_len]
    cdef DTYPE_t gallop = 1
    cdef DTYPE_t last = -1
    cdef DTYPE_t last_adj = -1
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]
    cdef DTYPE_t* lhs_adj_result_ptr = &adj_lhs_out[0]
    cdef DTYPE_t* rhs_adj_result_ptr = &adj_rhs_out[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:

        # Gallop to adjacent or equal value
        # if value_lhs < value_rhs - delta:
        # Gallop past the current element
        if (lhs_ptr[0] & args.mask) != (rhs_ptr[0] & args.mask):
            while lhs_ptr < end_lhs_ptr and ((lhs_ptr[0] & args.mask) + delta) < (rhs_ptr[0] & args.mask):
                lhs_ptr += (gallop * args.lhs_stride)
                gallop <<= 1
            lhs_ptr -= (gallop >> 1) * args.lhs_stride
            gallop = 1
            while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & args.mask) < ((lhs_ptr[0] & args.mask) + delta):
                rhs_ptr += (gallop * args.rhs_stride)
                gallop <<= 1
            rhs_ptr -= (gallop >> 1) * args.rhs_stride
            gallop = 1
            # Now lhs is at or before RHS - delta
            # RHS is 4, LHS is at most 3
        # Collect adjacent avalues
        if ((lhs_ptr[0] & args.mask) + delta) == ((rhs_ptr[0] & args.mask)):
            if (last_adj & args.mask) != (lhs_ptr[0] & args.mask):
                lhs_adj_result_ptr[0] = (lhs_ptr - &args.lhs[0]) / args.lhs_stride
                rhs_adj_result_ptr[0] = (rhs_ptr - &args.rhs[0]) / args.rhs_stride
                last_adj = lhs_ptr[0]
                lhs_adj_result_ptr += 1
                rhs_adj_result_ptr += 1
            lhs_ptr += args.lhs_stride
        # Now that we've reset, we just do the naive 2-ptr check
        # Then next loop we pickup on exponential search
        elif (lhs_ptr[0] & args.mask) < (rhs_ptr[0] & args.mask):
            lhs_ptr += args.lhs_stride
        elif (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr += args.rhs_stride
        else:
            # If here values equal, collect
            if (last & args.mask) != (lhs_ptr[0] & args.mask):
                lhs_result_ptr[0] = (lhs_ptr - &args.lhs[0]) / args.lhs_stride
                rhs_result_ptr[0] = (rhs_ptr - &args.rhs[0]) / args.rhs_stride
                last = lhs_ptr[0]
                lhs_result_ptr += 1
                rhs_result_ptr += 1
            rhs_ptr += args.rhs_stride

    adj_out_len[0] = lhs_adj_result_ptr - &adj_lhs_out[0]
    return lhs_result_ptr - &args.lhs_out[0]


def intersect(np.ndarray[DTYPE_t, ndim=1] lhs,
              np.ndarray[DTYPE_t, ndim=1] rhs,
              DTYPE_t mask=ALL_BITS,
              bint drop_duplicates=True):  # type: (np.ndarray, np.ndarray, DTYPE_t, bool)
    cdef np.uint64_t[:] lhs_out
    cdef np.uint64_t[:] rhs_out
    cdef DTYPE_t lhs_out_len = 0
    cdef DTYPE_t rhs_out_len = 0
    cdef intersect_args_t args

    if mask is None:
        mask = ALL_BITS
    if mask == 0:
        raise ValueError("Mask cannot be zero")

    args.mask = mask
    args.lhs = &lhs[0]
    args.rhs = &rhs[0]
    args.lhs_stride = lhs.strides[0] / sizeof(DTYPE_t)
    args.rhs_stride = rhs.strides[0] / sizeof(DTYPE_t)
    args.lhs_len = lhs.shape[0] * lhs.strides[0] / sizeof(DTYPE_t)
    args.rhs_len = rhs.shape[0] * rhs.strides[0] / sizeof(DTYPE_t)

    if drop_duplicates:
        lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
        rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
        args.lhs_out = &lhs_out[0]
        args.rhs_out = &rhs_out[0]
        with nogil:
            amt_written = _gallop_intersect_drop(args)
        return np.asarray(lhs_out)[:amt_written], np.asarray(rhs_out)[:amt_written]

    else:
        lhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
        rhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
        args.lhs_out = &lhs_out[0]
        args.rhs_out = &rhs_out[0]

        with nogil:
            _gallop_intersect_keep(args,
                                   &lhs_out_len, &rhs_out_len)
        lhs_out, rhs_out = np.asarray(lhs_out)[:lhs_out_len], np.asarray(rhs_out)[:rhs_out_len]
        return lhs_out, rhs_out


def adjacent(np.ndarray[DTYPE_t, ndim=1] lhs,
             np.ndarray[DTYPE_t, ndim=1] rhs,
             DTYPE_t mask=ALL_BITS):
    cdef np.uint64_t[:] lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef DTYPE_t amt_written = 0
    cdef DTYPE_t delta = 1
    if mask == 0:
        raise ValueError("Mask cannot be zero")
    if mask is None:
        mask = ALL_BITS
        delta = 1
    else:
        delta = (mask & -mask)  # lest significant set bit on mask

    with nogil:
        amt_written = _gallop_adjacent(&lhs[0], &rhs[0],
                                       lhs.shape[0], rhs.shape[0],
                                       &lhs_out[0], &rhs_out[0],
                                       mask, delta)
    return np.asarray(lhs_out[:amt_written]), np.asarray(rhs_out[:amt_written])


def intersect_with_adjacents(np.ndarray[DTYPE_t, ndim=1] lhs,
                             np.ndarray[DTYPE_t, ndim=1] rhs,
                             DTYPE_t mask=ALL_BITS):
    cdef np.uint64_t[:] lhs_out
    cdef np.uint64_t[:] rhs_out
    cdef np.uint64_t[:] adj_lhs_out
    cdef np.uint64_t[:] adj_rhs_out
    cdef np.uint64_t* adj_lhs_out_begin
    cdef np.uint64_t* adj_rhs_out_begin
    cdef intersect_args_t args
    cdef DTYPE_t delta = 1
    cdef DTYPE_t adj_out_len = 0

    if mask is None:
        mask = ALL_BITS
        delta = 1
    else:
        delta = (mask & -mask)  # lest significant set bit on mask
    if mask == 0:
        raise ValueError("Mask cannot be zero")

    args.mask = mask
    args.lhs = &lhs[0]
    args.rhs = &rhs[0]
    args.lhs_stride = lhs.strides[0] / sizeof(DTYPE_t)
    args.rhs_stride = rhs.strides[0] / sizeof(DTYPE_t)
    args.lhs_len = lhs.shape[0] * lhs.strides[0] / sizeof(DTYPE_t)
    args.rhs_len = rhs.shape[0] * rhs.strides[0] / sizeof(DTYPE_t)

    lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    adj_lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    adj_rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)

    args.lhs_out = &lhs_out[0]
    args.rhs_out = &rhs_out[0]
    adj_lhs_out_begin = &adj_lhs_out[0]
    adj_rhs_out_begin = &adj_rhs_out[0]
    print(f"Lhs len: {lhs.shape[0]} | Rhs len: {rhs.shape[0]}")
    with nogil:
        amt_written = _gallop_int_and_adj_drop(args, delta,
                                               adj_lhs_out_begin,
                                               adj_rhs_out_begin,
                                               &adj_out_len)
    return (np.asarray(lhs_out)[:amt_written], np.asarray(rhs_out)[:amt_written],
            np.asarray(adj_lhs_out)[:adj_out_len], np.asarray(adj_rhs_out)[:adj_out_len])



cdef DTYPE_t _int_w_index(intersect_args_t args,
                          DTYPE_t index_mask,
                          DTYPE_t* lhs_index,
                          DTYPE_t lhs_index_len,
                          DTYPE_t* rhs_index,
                          DTYPE_t rhs_index_len) nogil:
    """Two pointer intersect the index first THEN intersect the lhs / rhs within those indices."""
    cdef DTYPE_t* lhs_ptr = &args.lhs[0]
    cdef DTYPE_t* rhs_ptr = &args.rhs[0]
    cdef DTYPE_t* lhs_index_ptr = &lhs_index[0]
    cdef DTYPE_t* rhs_index_ptr = &rhs_index[0]
    cdef DTYPE_t* end_lhs_ptr = &args.lhs[args.lhs_len]
    cdef DTYPE_t* end_rhs_ptr = &args.rhs[args.rhs_len]
    cdef DTYPE_t* end_lhs_index_ptr = &lhs_index[lhs_index_len]
    cdef DTYPE_t* end_rhs_index_ptr = &rhs_index[rhs_index_len]
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]
    cdef DTYPE_t lhs_idx = 0 
    cdef DTYPE_t rhs_idx = 0
    cdef DTYPE_t lhs_curr_end = 0
    cdef DTYPE_t rhs_curr_end = 0
    cdef DTYPE_t index_lsb_mask = ~index_mask

    while lhs_index_ptr < end_lhs_index_ptr and rhs_index_ptr < end_rhs_index_ptr:
        if (lhs_index_ptr[0] & index_mask) < (rhs_index_ptr[0] & index_mask):
            lhs_index_ptr += 1
        elif (rhs_index_ptr[0] & index_mask) < (lhs_index_ptr[0] & index_mask):
            rhs_index_ptr += 1
        else:
            # Now two pointer intersect within lhs_index_ptr -> lhs_index_ptr[1]
            lhs_idx = lhs_index_ptr[0] & index_lsb_mask
            rhs_idx = rhs_index_ptr[0] & index_lsb_mask
            lhs_curr_end = (end_lhs_ptr - &args.lhs[0])
            rhs_curr_end = (end_rhs_ptr - &args.rhs[0])
            if lhs_index_ptr + 1 < end_lhs_index_ptr:
                lhs_curr_end = ((lhs_index_ptr + 1)[0] & index_lsb_mask)
            if rhs_index_ptr + 1 < end_rhs_index_ptr:
                rhs_curr_end = ((rhs_index_ptr + 1)[0] & index_lsb_mask)

            # Two pointer intesect between lhs_index_ptr and lhs_index_end w/ rhs_index_ptr and rhs_index_end
            while lhs_idx < lhs_curr_end and rhs_idx < rhs_curr_end:
                if (args.lhs[lhs_idx] & args.mask)  < (args.rhs[rhs_idx] & args.mask):
                    lhs_idx += 1
                elif (args.rhs[rhs_idx] & args.mask) < (args.lhs[lhs_idx] & args.mask):
                    rhs_idx += 1
                else:
                    lhs_result_ptr[0] = lhs_idx
                    rhs_result_ptr[0] = rhs_idx
                    lhs_result_ptr += 1
                    rhs_result_ptr += 1
                    lhs_idx += 1
                    rhs_idx += 1

            lhs_index_ptr += 1
            rhs_index_ptr += 1

    return lhs_result_ptr - &args.lhs_out[0]


cdef DTYPE_t _build_intersect_index(DTYPE_t* arr,
                                    DTYPE_t arr_len,
                                    DTYPE_t mask,
                                    DTYPE_t* idx_out) nogil:
    cdef DTYPE_t i = 0
    cdef DTYPE_t headerVal = 0xFFFFFFFFFFFFFFFF
    cdef DTYPE_t lastHeaderVal = 0
    cdef DTYPE_t* currIdxOut = &idx_out[0]
    for i in range(arr_len):
        headerVal = arr[i] & mask
        if headerVal != lastHeaderVal:
            currIdxOut[0] = (headerVal | i)
            currIdxOut += 1
            lastHeaderVal = headerVal
    return currIdxOut - &idx_out[0]



def int_w_index(np.ndarray[DTYPE_t, ndim=1] lhs,
                np.ndarray[DTYPE_t, ndim=1] rhs,
                np.ndarray[DTYPE_t, ndim=1] lhs_index,
                np.ndarray[DTYPE_t, ndim=1] rhs_index,
                DTYPE_t index_mask=ALL_BITS,
                DTYPE_t mask=ALL_BITS):
    cdef np.uint64_t[:] lhs_out
    cdef np.uint64_t[:] rhs_out
    cdef intersect_args_t args
    cdef DTYPE_t adj_out_len = 0
    cdef DTYPE_t* lhs_index_ptr = &lhs_index[0]
    cdef DTYPE_t* rhs_index_ptr = &rhs_index[0]
    cdef DTYPE_t lhs_index_len = lhs_index.shape[0]
    cdef DTYPE_t rhs_index_len = rhs_index.shape[0]
    
    lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    
    args.mask = mask
    args.lhs = &lhs[0]
    args.rhs = &rhs[0]
    args.lhs_len = lhs.shape[0] * lhs.strides[0] / sizeof(DTYPE_t)
    args.rhs_len = rhs.shape[0] * rhs.strides[0] / sizeof(DTYPE_t)
    args.lhs_out = &lhs_out[0]
    args.rhs_out = &rhs_out[0]

    with nogil:
        amt_written = _int_w_index(args,
                                   index_mask,
                                   lhs_index_ptr, lhs_index_len,
                                   rhs_index_ptr, rhs_index_len)

    return (np.asarray(lhs_out)[:amt_written], np.asarray(rhs_out)[:amt_written])




def build_intersect_index(np.ndarray[DTYPE_t, ndim=1] arr,
                          DTYPE_t mask=ALL_BITS):
    cdef np.uint64_t[:] idx_out = np.empty(arr.shape[0], dtype=np.uint64)
    cdef DTYPE_t amt_written = 0
    cdef DTYPE_t* arr_ptr = &arr[0]
    with nogil:
        amt_written = _build_intersect_index(arr_ptr, arr.shape[0], mask, &idx_out[0])
    return np.asarray(idx_out[:amt_written])
