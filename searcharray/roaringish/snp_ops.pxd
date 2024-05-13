# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef long long int64_t


ctypedef unsigned long long DTYPE_t
