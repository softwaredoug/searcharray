import numpy as np
import json
import os
from typing import Optional, List


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
                                   boundaries):
        metadata = {}
        for idx, (beg, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            offset = beg
            length = end - beg
            actual_id = ids[idx]
            metadata[actual_id] = {'offset': offset, 'length': length}
        arr = ArrayDict()
        arr.data = data
        arr.metadata = metadata
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
        self.metadata_file = root_filename + '.metadata.json'
        if arrays is not None:
            self._initialize_file(arrays)
            self._save_metadata()
        else:
            self._load_metadata()
            self._load_fp()

    def _initialize_file(self, arrays_dict):
        """Memmap underlying array in ArrayDict."""
        self.metadata = {}
        with open(self.filename, 'wb') as f:
            self.array.data.astype(self.dtype).tofile(f)

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.arrays.metadata, f)

    def _load_metadata(self):
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            pass

    def _load_fp(self):
        if self.fp is None:
            self.fp = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(self.fp.size,))

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
        self._save_metadata()
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