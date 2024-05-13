import numpy as np
from numpy.typing import NDArray


def popcount64(arr: NDArray[np.uint64]) -> NDArray[np.uint64]:
    ...


def popcount_reduce_at(ids: NDArray[np.uint64],
                       payload: NDArray[np.uint64],
                       out: NDArray[np.float64]) -> NDArray[np.float64]:
    ...
