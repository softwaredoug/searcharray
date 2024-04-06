"""Utilities for computing spans for position aware search with slop > 0."""
cimport numpy as np
import numpy as np
from enum import Enum

cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t


cdef extern from "stddef.h":
    # Get ctz and clz
    int __builtin_ctzll(unsigned long long x)
    int __builtin_clzll(unsigned long long x)
