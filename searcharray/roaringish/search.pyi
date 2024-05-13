import numpy as np
from numpy.typing import NDArray
from typing import Tuple


ALL_BITS = np.uint64(-1)


def binary_search(arr: NDArray[np.uint64],
                  target: np.uint64,
                  mask: np.uint64 = ALL_BITS,
                  start: np.uint64 = np.uint64(0)) -> Tuple[np.uint64, bool]:
    ...


def galloping_search(arr: NDArray[np.uint64],
                     target: np.uint64,
                     mask: np.uint64 = ALL_BITS,
                     start: np.uint64 = np.uint64(0)) -> Tuple[np.uint64, bool]:
    ...


def count_odds(lhs: NDArray[np.uint64],
               rhs: NDArray[np.uint64]) -> np.uint64:
    ...
