import numpy as np
import os
import reprlib
from typing import Optional, List, Dict


def create_filename(data_dir: str):
    """Create a filename for the memory-mapped array"""
    # Ls path and get the number of files
    files = os.listdir(data_dir)
    num_files = len(files)
    return os.path.join(data_dir, f'{num_files}')


class ArrayDict:
    """An array of ints to np.array as contiguous memory"""

    def __init__(self):
        self.dtype = np.uint64
        self.data = np.ndarray(0, dtype=self.dtype)
        self.metadata = {}

    def __repr__(self):
        repr_obj = reprlib.Repr()
        return f'ArrayDict({repr_obj.repr(self.metadata)})'

    @staticmethod
    def from_arrays(arrays: List[np.ndarray]):
        arr = ArrayDict()
        metadata = {}
        total_len = 0
        for i, array in enumerate(arrays):
            metadata[i] = {'offset': total_len, 'length': array.size}
            total_len += array.size
        data = np.concatenate(arrays)
        arr.data = data
        arr.metadata = metadata
        return arr

    @staticmethod
    def from_array_with_boundaries(data: np.ndarray,
                                   ids: np.ndarray,
                                   boundaries: np.ndarray):
        metadata = {}
        for idx, (beg, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            offset = int(beg)
            length = int(end - beg)
            actual_id = ids[idx]
            metadata[int(actual_id)] = {'offset': offset, 'length': length}
        arr = ArrayDict()
        arr.data = data
        arr.metadata = metadata
        return arr

    @staticmethod
    def concat(lhs: 'ArrayDict', rhs: 'ArrayDict', sort=True):
        arr = ArrayDict()
        metadata: Dict[int, Dict[str, int]] = {}
        lst_of_arrays: List[np.ndarray] = []

        curr_offset = 0
        last_offset = 0

        fetched_keys = set()
        for key, value in lhs.items():
            fetched_keys.add(key)
            all_for_key = [value]
            curr_offset += value.size
            if key in rhs.metadata:
                rhs_value = rhs[key]
                all_for_key.append(rhs_value)
                curr_offset += rhs_value.size
            metadata[key] = {'offset': last_offset, 'length': curr_offset - last_offset}
            last_offset = curr_offset
            lst_of_arrays.append(np.concatenate(all_for_key))

        for key, value in rhs.items():
            if key not in fetched_keys:
                curr_offset += value.size
                lst_of_arrays.append(value)
                metadata[key] = {'offset': last_offset, 'length': curr_offset - last_offset}
                last_offset = curr_offset

        arr.metadata = metadata
        for curr_arr in lst_of_arrays:
            if sort:
                curr_arr.sort()

        if len(lst_of_arrays) > 0:
            arr.data = np.concatenate(lst_of_arrays)
        return arr

    def __getitem__(self, key):
        key = int(key)
        if key in self.metadata:
            offset = self.metadata[key]['offset']
            length = self.metadata[key]['length']
            return self.data[offset:offset + length]
        else:
            raise KeyError(f'Key {key} not found.')

    def __setitem__(self, key, value):
        key = int(key)
        if value.dtype != self.dtype:
            raise ValueError(f'Value must be of type {self.dtype}')
        offset = self.data.size
        length = value.size
        self.metadata[key] = {'offset': offset, 'length': length}
        self.data = np.append(self.data, value)

    def __delitem__(self, key):
        key = int(key)
        if key in self.metadata:
            del self.metadata[key]
        else:
            raise KeyError(f'Key {key} not found.')

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        yield from self.metadata

    def __contains__(self, key):
        return key in self.metadata

    def items(self):
        for key, value in self.metadata.items():
            yield key, self.data[value['offset']:value['offset'] + value['length']]

    def keys(self):
        return self.metadata.keys()

    def compact(self):
        """Compact the data array"""
        new_data = np.ndarray(sum([value['length'] for value in self.metadata.values()]), dtype=self.dtype)
        offset = 0
        for value in self.metadata.values():
            new_data[offset:offset + value['length']] = self.data[value['offset']:value['offset'] + value['length']]
            value['offset'] = offset
            offset += value['length']
        self.data = new_data


class MemoryMappedArrays:
    def __init__(self,
                 data_dir: str,
                 arrays: Optional[ArrayDict] = None):
        root_filename = create_filename(data_dir)
        self.filename = root_filename + '.dat'
        self.fp = None
        if arrays is not None:
            self.arrays = arrays
            self._initialize_file()
        else:
            self._load_arrays()

    def _initialize_file(self):
        """Memmap underlying array in ArrayDict."""
        with open(self.filename, 'wb') as f:
            self.arrays.data.tofile(f)

    def _load_arrays(self):
        """Memmap underlying array in ArrayDict from self.filename."""
        self.arrays.data = np.memmap(self.filename, dtype=self.arrays.dtype, mode='r+')

    def __getitem__(self, key):
        # Use metadatat to get the offset and length of the array
        return self.arrays[key]

    def __setitem__(self, key, value):
        """Append to file and update metadata. Not fast."""
        self.arrays[key] = value
        self._save_metadata()
        self.fp = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(self.fp.size + value.size,))

    def __delitem__(self, key):
        length = self.arrays[key].size
        del self.arrays[key]
        self.fp = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(self.fp.size - length,))

    def __len__(self):
        return len(self.arrays)

    def __iter__(self):
        yield from self.arrays

    def __contains__(self, key):
        return key in self.arrays

    def items(self):
        yield from self.arrays.items()

    def keys(self):
        return self.arrays.keys()

    def __getstate__(self):
        """Return arrays without the numpy array."""
        # Create arrays without data
        arrays = ArrayDict()
        arrays.metadata = self.arrays.metadata
        return {'arrays': arrays,
                'filename': self.filename}

    def __setstate__(self, state):
        self.filename = state['filename']
        self.arrays = state['arrays']
        self._load_arrays()
