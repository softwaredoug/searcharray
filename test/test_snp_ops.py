from typing import Tuple
import numpy as np
import sortednp as snp
import pytest
from searcharray.roaringish.snp_ops import binary_search, galloping_search, intersect, adjacent, unique, merge
from test_utils import w_scenarios
from test_utils import Profiler, profile_enabled


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


@pytest.mark.parametrize("array", [u64([0, 3, 11, 23, 32, 36, 41, 42])])
@pytest.mark.parametrize("algorithm", [binary_search, galloping_search])
def test_search_start_at_end(array, algorithm):
    start = len(array) - 2
    idx, found = algorithm(array, array[-1], start=start)
    assert idx == u64(len(array) - 1)
    assert found


mask_search_scenarios = [
    [0x0000000100000000],
    [0x0000001100000000, 0x0000011100000000],
    [0x0000001100000001, 0x0000001100000011, 0x0000011100000000]
]


@w_search_scenarios(mask_search_scenarios)
@pytest.mark.parametrize("algorithm", [binary_search, galloping_search])
@pytest.mark.parametrize("mask", [np.uint64(0xFFFFFFFF00000000)])
def test_search_masks(mask, algorithm, array: np.ndarray, target: np.uint64, expected: Tuple[np.uint64, bool]):
    # Reset expected to first in array with mask
    for idx, val in enumerate(array):
        if val & mask == target & mask:
            expected = (np.uint64(idx), True)
            break

    idx, found = algorithm(array, target, mask)
    if expected[1]:
        assert array[idx] & mask == target & mask
        assert idx == expected[0]
        assert found == expected[1]
    else:
        assert not found


intersect_scenarios = {
    "base": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([2, 4, 6, 8, 10]),
        "mask": None,
        "expected": u64arr([2, 4, 6, 8, 10])
    },
    "base_dups": {
        "lhs": u64([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        "rhs": u64([1, 2, 2, 10]),
        "mask": None,
        "expected": u64arr([1, 2])
    },
    "lsbs_differ": {
        "lhs": u64([0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF]),
        "rhs": u64([0x2F, 0x4F, 0x6F, 0x8F, 0xAF]),
        "mask": None,
        "expected": u64arr([0x2F, 0x4F, 0x6F, 0x8F, 0xAF])
    },
    "rhs_first": {
        "lhs": u64([2, 4, 6, 8, 10]),
        "rhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "mask": None,
        "expected": u64arr([2, 4, 6, 8, 10])
    },
    "base_mask": {
        "lhs": u64([0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF]),
        "rhs": u64([0x2F, 0x4F, 0x6F, 0x8F, 0xAF]),
        "mask": u64(0xF0),
        "expected": u64arr([0x20, 0x40, 0x60, 0x80, 0xA0])
    },
    # Cases from comparing random data to sortednp library
    "rand1": {
        "lhs": u64([10, 11, 13, 19, 21, 26, 26, 30, 41, 43]),
        "rhs": u64([2, 5, 19, 20, 46, 46, 46, 55, 56, 57]),
        "mask": None,
        "expected": u64([19])
    },
    "intersect_at_end": {
        "lhs": u64([9, 25, 28, 31, 31, 32, 38, 39, 42]),
        "rhs": u64([0, 3, 11, 23, 32, 36, 41, 42]),
        "mask": None,
        "expected": u64([32, 42])
    },
}


@w_scenarios(intersect_scenarios)
def test_intersect(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


@w_scenarios(intersect_scenarios)
def test_intersect_keep_both(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask, drop_duplicates=False)
    expected_lhs = np.argwhere(np.in1d(lhs, rhs)).flatten()
    expected_rhs = np.argwhere(np.in1d(rhs, lhs)).flatten()
    assert np.all(lhs_idx == expected_lhs)
    assert np.all(rhs_idx == expected_rhs)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_same_as_numpy(seed):
    np.random.seed(seed)
    rand_arr_1 = np.random.randint(0, 500, 100, dtype=np.uint64)
    rand_arr_2 = np.random.randint(0, 500, 100, dtype=np.uint64)
    rand_arr_1.sort()
    rand_arr_2.sort()

    expected_result = np.intersect1d(rand_arr_1, rand_arr_2)

    lhs_idx, rhs_idx = intersect(rand_arr_1, rand_arr_2)
    result = rand_arr_1[lhs_idx]
    assert np.all(result == expected_result)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_masked_intersect(benchmark):
    profiler = Profiler(benchmark)

    rand_arr_1 = np.random.randint(0, 50000, 1000000, dtype=np.uint64)
    rand_arr_2 = np.random.randint(0, 50000, 1000000, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF00000000)
    rand_arr_1.sort()
    rand_arr_2.sort()

    def with_snp():
        snp.intersect(rand_arr_1 << 16, rand_arr_2 << 16, indices=True, duplicates=snp.DROP)

    def with_snp_ops():
        intersect(rand_arr_1, rand_arr_2, mask)

    def intersect_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()

    profiler.run(intersect_many)


@pytest.mark.parametrize("array", [u64([0, 0, 11, 11, 11, 36, 41, 42])])
def test_unique(array):
    expected = np.unique(array)
    result = unique(array)
    assert np.all(result == expected)


@pytest.mark.parametrize("array,shift", [(u64([0xEE00, 0xFF00, 0xFF01]), 8)])
def test_unique_shifted(array, shift):
    expected = np.unique(array >> shift)
    result = unique(array, shift)
    assert np.all(result == expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_unique_matches_snp(seed):
    np.random.seed(seed)
    rand_arr_1 = np.random.randint(0, 500, 100, dtype=np.uint64)
    rand_arr_1.sort()
    expected = np.unique(rand_arr_1)
    result = unique(rand_arr_1)
    assert np.all(result == expected)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_unique(benchmark):
    rand_arr_1 = np.random.randint(0, 50, 1000000, dtype=np.uint64)
    rand_arr_1.sort()

    def with_snp():
        snp.intersect(rand_arr_1, rand_arr_1, duplicates=snp.DROP)

    def with_np():
        np.unique(rand_arr_1)

    def with_snp_ops():
        unique(rand_arr_1)

    def unique_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            with_np()

    profiler = Profiler(benchmark)
    profiler.run(unique_many)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_unique_shifted(benchmark):
    rand_arr_1 = np.random.randint(0, 50, 1000000, dtype=np.uint64)
    rand_arr_1.sort()

    def with_snp():
        shifted = rand_arr_1 >> 16
        snp.intersect(shifted, shifted, duplicates=snp.DROP)

    def with_np():
        np.unique(rand_arr_1 >> 16)

    def with_snp_ops():
        unique(rand_arr_1, 16)

    def unique_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            with_np()

    profiler = Profiler(benchmark)
    profiler.run(unique_many)


adj_scenarios = {
    "base": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([2, 4, 6, 8, 10]),
        "mask": None,
        "lhs_idx_expected": u64arr([0, 2, 4, 6, 8]),
        "rhs_idx_expected": u64arr([0, 1, 2, 3, 4])
    },
    "start_0s": {
        "lhs": u64([0, 0, 3]),
        "rhs": u64([0, 0, 1, 8, 10]),
        "mask": None,
        "lhs_idx_expected": u64arr([0]),
        "rhs_idx_expected": u64arr([2])
    },
    "start_0s_2": {
        "lhs": u64([0, 2, 3]),
        "rhs": u64([0, 1, 2, 8, 10]),
        "mask": None,
        "lhs_idx_expected": u64arr([0]),
        "rhs_idx_expected": u64arr([1])
    },
    "base_masked": {
        "lhs": u64([0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF]),
        "rhs": u64([0x2F, 0x4F, 0x6F, 0x8F, 0xAF]),
        "mask": 0xF0,
        "lhs_idx_expected": u64arr([0, 2, 4, 6, 8]),
        "rhs_idx_expected": u64arr([0, 1, 2, 3, 4])
    },
    "rhs_0": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([0, 3]),
        "mask": None,
        "lhs_idx_expected": u64arr([1]),
        "rhs_idx_expected": u64arr([1])
    },
    "many_adjacent": {
        "lhs": u64([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        "rhs": u64([2, 3]),
        "mask": None,
        "lhs_idx_expected": u64arr([0, 5]),  # As we drop dups
        "rhs_idx_expected": u64arr([0, 1])
    },
    "trouble_scen": {
        "lhs": u64([1, 274877906945, 549755813889, 824633720833]),
        "rhs": u64([6, 137438953474, 274877906950, 412316860418]),
        "mask": 0xfffffffffffc0000,
        "lhs_expected": u64arr([]),
        "rhs_expected": u64arr([])
    }
}


@w_scenarios(adj_scenarios)
def test_adjacent(lhs, rhs, mask, lhs_idx_expected, rhs_idx_expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx = adjacent(lhs, rhs, mask)
    assert np.all(lhs_idx == lhs_idx_expected)
    assert np.all(rhs_idx == rhs_idx_expected)


merge_scenarios = {
    "base": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([2, 4, 6, 8, 10]),
        "expected": u64arr([1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10])
    },
    "rhs_empty": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([]),
        "expected": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    },
}


@w_scenarios(merge_scenarios)
def test_merge(lhs, rhs, expected):
    result = merge(lhs, rhs)
    assert np.all(result == expected)


@w_scenarios(merge_scenarios)
def test_merge_w_drop(lhs, rhs, expected):
    result = merge(lhs, rhs, drop_duplicates=True)
    assert np.all(result == np.sort(np.unique(expected)))


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_merge(benchmark):
    rand_arr_1 = np.random.randint(0, 50, 1000000, dtype=np.uint64)
    rand_arr_1.sort()
    rand_arr_2 = np.random.randint(0, 50, 1000000, dtype=np.uint64)
    rand_arr_2.sort()

    def with_snp():
        snp.merge(rand_arr_1, rand_arr_2)

    def with_snp_ops():
        merge(rand_arr_1, rand_arr_2)

    def merge_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()

    profiler = Profiler(benchmark)
    profiler.run(merge_many)
