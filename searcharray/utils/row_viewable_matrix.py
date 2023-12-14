import numbers
from scipy.sparse import csr_matrix
import numpy as np


def rowwise_eq(mat: csr_matrix, other: csr_matrix) -> np.ndarray:
    """Check equals on a row-by-row basis."""
    if mat.shape != other.shape:
        return False
    # Subtracting csr mats faster than eq, because
    # eq seems to construct a coo matrix under the hood?
    eq_mat = mat - other
    num_cols_eq_per_row = np.sum(eq_mat, axis=1)
    row_eq = (num_cols_eq_per_row == 0).flatten()
    return np.squeeze(np.asarray(row_eq))


class RowViewableMatrix:
    """A slicable matrix that can return views without copying."""

    def __init__(self, csr_mat: csr_matrix, rows: np.ndarray = None):
        self.mat = csr_mat
        self.col_cache = {}
        self.cols_cached = []
        if rows is None:
            self.rows = np.arange(self.mat.shape[0])
        elif isinstance(rows, numbers.Integral):
            self.rows = np.array([rows])
        else:
            self.rows = rows

    def slice(self, keys):
        return RowViewableMatrix(self.mat, self.rows[keys])

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
        return RowViewableMatrix(self.mat.copy(), self.rows.copy())

    def sum(self, axis=0):
        return self.mat[self.rows].sum(axis=axis)

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
        return self.mat.data.nbytes + \
            self.mat.indptr.nbytes + \
            self.mat.indices.nbytes + \
            self.rows.nbytes

    @property
    def shape(self):
        return (len(self.rows), self.mat.shape[1])

    def resize(self, shape):
        self.mat.resize(shape)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"RowViewableMatrix({repr(self.mat)}, {repr(self.rows)})"

    def __str__(self):
        return f"RowViewableMatrix({str(self.mat)}, {str(self.rows)})"

    def __eq__(self, other):
        return rowwise_eq(self.mat[self.rows], other.mat[other.rows])
