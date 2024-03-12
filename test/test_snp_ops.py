from typing import Tuple
import numpy as np
import pytest
from searcharray.utils.snp_ops import binary_search, galloping_search


def u64arr(lst):
    return np.asarray(lst, dtype=np.uint64)


def u64(v):
    if isinstance(v, list):
        return u64arr([np.uint64(x) for x in v])
    return np.uint64(v)


search_scenarios = [
    [1],
    [1, 2, 500, 1000],
    [1, 2, 500, 1000, 1001, 1002],
    [100000, 100001],
]


def w_search_scenarios(lst):
    """Decorate with array and every possible target."""
    params = []
    for arr in lst:
        for idx in range(len(arr)):
            np_arr = u64(arr)
            # assert sorted
            assert np.all(np_arr[:-1] <= np_arr[1:])
            params.append((u64(arr),
                           u64(arr[idx]),
                           (u64(idx), True)))
        # Add one before and one after
        params.append((u64(arr), u64(arr[0] - 1), (u64(0), False)))
        params.append((u64(arr), u64(arr[-1] + 1), (u64(len(arr)), False)))
    return pytest.mark.parametrize(
        "array,target,expected", params
    )


@w_search_scenarios(search_scenarios)
@pytest.mark.parametrize("algorithm", [binary_search, galloping_search])
def test_search(algorithm, array: np.ndarray, target: np.uint64, expected: Tuple[np.uint64, bool]):
    idx, found = algorithm(array, target)
    if expected[1]:
        assert idx == expected[0]
        assert found == expected[1]
    else:
        assert not found


mask_search_scenarios = [
    [0x0000000100000000],
    [0x0000001100000000, 0x0000011100000000],
    [0x0000001100000001, 0x0000001100000011, 0x0000011100000000]
]


@w_search_scenarios(mask_search_scenarios)
@pytest.mark.parametrize("algorithm", [binary_search, galloping_search])
@pytest.mark.parametrize("mask", [np.uint64(0xFFFFFFFF00000000)])
def test_search_masks(mask, algorithm, array: np.ndarray, target: np.uint64, expected: Tuple[np.uint64, bool]):
    idx, found = algorithm(array, target, mask)
    if expected[1]:
        assert array[idx] & mask == target & mask
        assert idx == expected[0]
        assert found == expected[1]
    else:
        assert not found
