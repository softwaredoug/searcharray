import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def merge(lhs: NDArray[np.uint64],
          rhs: NDArray[np.uint64],
          drop_duplicates: bool = False):
    ...


def sort_merge_counts(lhs_ids: NDArray[np.uint64],
                      lhs_counts: NDArray[np.float32],
                      rhs_ids: NDArray[np.uint64],
                      rhs_counts: NDArray[np.float32]) -> Tuple[NDArray[np.uint64], NDArray[np.float32]]:
    ...
