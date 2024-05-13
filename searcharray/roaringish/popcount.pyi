import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def popcount64(arr: NDArray[np.uint64]) -> NDArray[np.uint64]:
    ...


def popcount_reduce_at(ids: NDArray[np.uint64],
                       payload: NDArray[np.uint64],
                       out: NDArray[np.float64]) -> NDArray[np.float64]:
    ...


def popcount64_reduce(arr: NDArray[np.uint64],
                      key_shift: np.uint64,
                      value_mask: np.uint64) -> Tuple[NDArray[np.uint64],
                                                      NDArray[np.uint64]]:
    ...
