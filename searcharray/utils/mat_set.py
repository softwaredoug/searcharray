import numpy as np
import pandas as pd
import numbers


class SparseMatSetBuilder:

    def __init__(self):
        self.cols = []
        self.rows = [0]

    def append(self, cols):
        self.cols.extend(cols)
        self.rows.append(len(self.cols))
        return 0

    def concat(self, other: 'SparseMatSetBuilder'):
        self.rows.extend([row + len(self.cols) for row in other.rows[1:]])
        self.cols.extend(other.cols)
        return 0

    def build(self):
        return SparseMatSet(cols=np.asarray(self.cols, dtype=np.uint32),
                            rows=np.asarray(self.rows, dtype=np.uint32))


class SparseMatSet:
    """Sparse matrix that only stores the set of row/col indices that are set to 1."""

    def __init__(self, cols=None, rows=None):
        if rows is None:
            rows = np.asarray([0], dtype=np.uint32)
            cols = np.asarray([], dtype=np.uint32)
        self.cols = cols.astype(np.uint32)  # col indices.
        self.rows = rows.astype(np.uint32)  # indices into cols
        assert self.rows[-1] == len(self.cols)

    def __getitem__(self, key):
        # Iterate keys
        beg_keys = self.rows[:-1][key]
        end_keys = self.rows[1:][key]

        if not isinstance(beg_keys, np.ndarray):
            beg_keys = np.asarray([beg_keys])
            end_keys = np.asarray([end_keys])

        cols = [self.cols[beg:end] for beg, end in zip(beg_keys, end_keys)]
        rows = [0] + [len(row) for row in cols]
        rows = np.asarray(rows).flatten()
        rows = np.cumsum(rows)
        try:
            cols = np.concatenate(cols)
        except ValueError:
            cols = np.asarray([], dtype=np.uint32)
        return SparseMatSet(cols, rows)

    def append(self, other: 'SparseMatSet'):
        self.cols = np.concatenate([self.cols, other.cols])
        # Increment other rows
        self.rows = np.concatenate([self.rows, self.rows[-1] + other.rows])
        return self

    def ensure_capacity(self, row):
        if row >= len(self):
            append_amt = row - (len(self.rows) - 1) + 1
            new_row_ptrs = [len(self.cols)] * append_amt
            self.rows = np.concatenate([self.rows, new_row_ptrs])

    def set_cols(self, row, cols, overwrite=False):
        row = np.int32(row)
        self.ensure_capacity(row)

        cols_for_row = self.rows[:-1][row]
        cols_for_row_next = self.rows[1:][row]
        front_cols = np.asarray([], dtype=np.int64)
        trailing_cols = np.asarray([], dtype=np.int64)
        if row > 0:
            front_cols = self.cols[:cols_for_row]
        if cols_for_row_next != self.rows[-1]:
            trailing_cols = self.cols[cols_for_row_next:]

        existing_set_cols = self.cols[cols_for_row:cols_for_row_next]
        cols_added = np.int32(len(np.setdiff1d(cols, existing_set_cols)))
        if not overwrite:
            existing_set_cols = np.unique(np.concatenate([cols, existing_set_cols]))

            self.cols = np.concatenate([front_cols, existing_set_cols, trailing_cols], dtype=np.int64)
        else:
            cols_added = np.int32(len(cols) - len(existing_set_cols))
            self.cols = np.concatenate([front_cols, cols, trailing_cols], dtype=np.int64)

        if cols_added < 0:
            # TODO some casting nonsense makes this necessary
            self.rows[row + 1:] -= np.abs(cols_added)
        else:
            self.rows[row + 1:] += cols_added

    def __setitem__(self, index, value):
        if isinstance(index, numbers.Integral):
            if len(value.shape) == 1:
                value = value.reshape(1, -1)
            set_rows, set_cols = value.nonzero()
            if not (value[set_rows, set_cols] == 1).all():
                raise ValueError("This sparse matrix only supports setting 1")
            self.set_cols(index, set_cols, overwrite=True)

        # Multidimensional indexing
        elif isinstance(index, tuple):
            row, col = index
            if value != 1:
                raise ValueError("This sparse matrix only supports setting 1")
            self.set_cols(row, np.asarray([col]))
        # Multiple rows
        elif pd.api.types.is_list_like(index):
            if len(index) == len(value):
                for idx, val in zip(index, value):
                    self[idx] = val
            elif len(value) == 1:
                for idx in index:
                    self[idx] = value
            else:
                raise ValueError("Index and value must be same length")

    def copy(self):
        return SparseMatSet(self.cols.copy(), self.rows.copy())

    @property
    def nbytes(self):
        return self.cols.nbytes + self.rows.nbytes

    @property
    def shape(self):
        rows = len(self.rows) - 1
        cols = 0
        if len(self.cols) > 0:
            cols = np.max(self.cols)
        return (rows, cols)

    def num_cols_per_row(self):
        return np.diff(self.rows)

    def __len__(self):
        return len(self.rows) - 1

    def __eq__(self, other):
        return np.all(self.rows == other.rows) and np.all(self.cols == other.cols)

    def __repr__(self):
        return f"SparseMatSet(shape={self.shape})"

    def __str__(self):
        as_str = [""]
        for idx, (row, row_next) in enumerate(zip(self.rows, self.rows[1:])):
            as_str.append(f"{idx}: {self.cols[row:row_next]}")
        return "\n".join(as_str)
