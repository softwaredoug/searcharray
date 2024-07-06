import numpy as np
import json
import os


def create_filename(data_dir: str):
    """Create a filename for the memory-mapped array"""
    # Ls path and get the number of files
    files = os.listdir(data_dir)
    num_files = len(files)
    return os.path.join(data_dir, f'{num_files}')


class MemoryMappedArrays:
    def __init__(self,
                 data_dir: str,
                 arrays: dict[int, np.ndarray] | None = None):
        root_filename = create_filename(data_dir)
        self.filename = root_filename + '.dat'
        self.fp = None
        self.metadata_file = root_filename + '.metadata.json'
        self.metadata: dict[int, dict] = {}
        self.dtype = np.uint64
        if arrays is not None:
            self._initialize_file(arrays)
            self._save_metadata()
        else:
            self._load_metadata()
            self._load_fp()

    def _initialize_file(self, arrays_dict):
        total_len = 0
        with open(self.filename, 'wb') as f:
            for key, array in arrays_dict.items():
                offset = f.tell()
                length = array.size
                total_len += length
                key = int(key)
                self.metadata[key] = {'offset': offset, 'length': length}
                array.astype(self.dtype).tofile(f)
        # Load mmmap array
        self.fp = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(total_len,))

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

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
        key = int(key)
        if key in self.metadata:
            offset = self.metadata[key]['offset']
            length = self.metadata[key]['length']
            return np.memmap(self.filename, dtype=self.dtype, mode='r+', offset=offset, shape=(length,))
        else:
            raise KeyError(f'Key {key} not found.')

    def __setitem__(self, key, value):
        """Append to file and update metadata"""
        key = int(key)
        if key in self.metadata:
            raise KeyError(f'Key {key} already exists.')
        with open(self.filename, 'ab') as f:
            offset = f.tell()
            length = value.size
            self.metadata[key] = {'offset': offset, 'length': length}
            value.astype(self.dtype).tofile(f)
        self._save_metadata()
        self.fp = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(self.fp.size + length,))

    def __delitem__(self, key):
        key = int(key)
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
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
            yield key, np.memmap(self.filename, dtype=self.dtype, mode='r', offset=value['offset'], shape=(value['length'],))
