import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

ALL_BITS = np.uint64(-1)


def intersect(lhs: NDArray[np.uint64],
              rhs: NDArray[np.uint64],
              mask: Optional[np.uint64] = ALL_BITS,
              drop_duplicates: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    ...


def adjacent(lhs: NDArray[np.uint64],
             rhs: NDArray[np.uint64],
             mask: np.uint64 = ALL_BITS) -> Tuple[np.ndarray, np.ndarray]:
    ...


def intersect_with_adjacents(lhs: NDArray[np.uint64],
                             rhs: NDArray[np.uint64],
                             mask: np.uint64 = ALL_BITS) -> Tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
    ...


def int_w_index(lhs: NDArray[np.uint64],
                rhs: NDArray[np.uint64],
                lhs_index: NDArray[np.uint64],
                rhs_index: NDArray[np.uint64],
                index_mask: np.uint64 = ALL_BITS,
                mask: np.uint64 = ALL_BITS) -> Tuple[np.ndarray, np.ndarray]:
    ...


def build_intersect_index(arr: NDArray[np.uint64],
                          mask: np.uint64) -> NDArray[np.uint64]:
    ...
