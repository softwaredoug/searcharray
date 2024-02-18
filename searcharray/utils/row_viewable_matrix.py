import numbers
import numpy as np
from searcharray.utils.mat_set import SparseMatSet
from typing import Optional, Union, Dict, List


def rowwise_eq(mat: SparseMatSet, other: SparseMatSet) -> Union[bool, np.ndarray]:
    """Check equals on a row-by-row basis."""
    if len(mat) != len(other):
        return False
    row_eq = np.zeros(mat.shape[0], dtype=np.dtype('bool'))
    for row_idx in range(len(mat)):
        if np.all(mat[row_idx] == other[row_idx]):
            row_eq[row_idx] = True
    return row_eq


class RowViewableMatrix:
    """A slicable matrix that can return views without copying."""

    def __init__(self, mat: SparseMatSet, rows: Optional[np.ndarray] = None, subset=False):
        self.mat = mat
        self.col_cache: Dict[int, np.ndarray] = {}
        self.cols_cached: List[int] = []
        if rows is None:
            self.rows = np.arange(self.mat.shape[0])
        elif isinstance(rows, numbers.Integral):
            self.rows = np.array([rows])
        else:
            self.rows = rows
        self.subset = subset

    def slice(self, keys):
        return RowViewableMatrix(self.mat, self.rows[keys], subset=True)

    def __setitem__(self, keys, values):
        # Replace nan with 0
        self.col_cache = {}
        self.cols_cached = []
        actual_keys = self.rows[keys]
        if isinstance(actual_keys, numbers.Integral):
            self.mat[actual_keys] = values
        elif len(actual_keys) > 0:
            self.mat[actual_keys] = values

    def copy_row_at(self, row):
        return self.mat[self.rows[row]]

    def copy(self):
        return RowViewableMatrix(self.mat.copy(), self.rows.copy(), subset=self.subset)

    def cols_per_row(self):
        return self.mat[self.rows].num_cols_per_row()

    def copy_col_at(self, col):
        if col not in self.col_cache:
            self.col_cache[col] = self.mat[self.rows, col]
            self.cols_cached.append(col)
            if len(self.cols_cached) > 10:
                del self.col_cache[self.cols_cached.pop(0)]
        return self.col_cache[col]

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self.copy_row_at(key)
        else:
            return self.slice(key)

    @property
    def nbytes(self):
        return self.mat.nbytes + \
            self.rows.nbytes

    @property
    def shape(self):
        return (len(self.rows), self.mat.shape[1])

    def resize(self, shape):
        self.mat.ensure_capacity(shape[0] - 1)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"RowViewableMatrix({repr(self.mat)}, {repr(self.rows)})"

    def __str__(self):
        return f"RowViewableMatrix({str(self.mat)}, {str(self.rows)})"

    def __eq__(self, other):
        return rowwise_eq(self.mat[self.rows], other.mat[other.rows])
