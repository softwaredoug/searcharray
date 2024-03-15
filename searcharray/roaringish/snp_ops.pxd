# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t


ctypedef unsigned long long DTYPE_t


cimport numpy as np


cdef void _galloping_search(DTYPE_t[:] array,
                            DTYPE_t target,
                            DTYPE_t mask,
                            np.intp_t* i,
                            np.intp_t len)
