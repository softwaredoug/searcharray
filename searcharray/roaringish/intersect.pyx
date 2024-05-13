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


cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport DTYPE_t, int64_t

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
    cdef DTYPE_t delta = 1
    cdef DTYPE_t last = -1
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:

        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & args.mask) < (rhs_ptr[0] &args. mask):
            lhs_ptr+= (delta * args.lhs_stride)
            delta *= 2
        lhs_ptr -= ((delta // 2) * args.lhs_stride)
        delta = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr+= (delta * args.rhs_stride)
            delta *= 2
        rhs_ptr -= ((delta // 2) * args.rhs_stride)
        delta = 1

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
    cdef DTYPE_t delta = 1
    cdef DTYPE_t target = -1
    # cdef np.uint64_t[:] lhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    # cdef np.uint64_t[:] rhs_out = np.empty(max(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef DTYPE_t* lhs_result_ptr = &args.lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &args.rhs_out[0]

    while lhs_ptr < end_lhs_ptr and rhs_ptr < end_rhs_ptr:
        # Gallop past the current element
        while lhs_ptr < end_lhs_ptr and (lhs_ptr[0] & args.mask) < (rhs_ptr[0] & args.mask):
            lhs_ptr += (delta * args.lhs_stride)
            delta <<= 1
        lhs_ptr -= ((delta >> 1) * args.lhs_stride)
        delta = 1
        while rhs_ptr < end_rhs_ptr and (rhs_ptr[0] & args.mask) < (lhs_ptr[0] & args.mask):
            rhs_ptr += (delta * args.rhs_stride)
            delta <<= 1
        rhs_ptr -= ((delta >> 1) * args.rhs_stride)
        delta = 1


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
    cdef DTYPE_t lhs_delta = 1
    cdef DTYPE_t rhs_delta = 1
    cdef DTYPE_t last = -1
    cdef DTYPE_t* lhs_result_ptr = &lhs_out[0]
    cdef DTYPE_t* rhs_result_ptr = &rhs_out[0]
    
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

    return lhs_result_ptr - &lhs_out[0]



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
    args.rhs_stride =  rhs.strides[0] / sizeof(DTYPE_t)
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
             DTYPE_t mask=ALL_BITS):  # type: (np.ndarray, np.ndarray, DTYPE_t)
    cdef np.uint64_t[:] lhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef np.uint64_t[:] rhs_out = np.empty(min(lhs.shape[0], rhs.shape[0]), dtype=np.uint64)
    cdef DTYPE_t amt_written = 0
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
    return lhs_out[:amt_written], rhs_out[:amt_written]
