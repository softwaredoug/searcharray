from typing import Tuple
import numpy as np
import sortednp as snp
import pytest
from searcharray.roaringish.search import binary_search, galloping_search, count_odds
from searcharray.roaringish.unique import unique
from searcharray.roaringish.merge import merge
from searcharray.roaringish.intersect import intersect, adjacent, intersect_with_adjacents
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
    "trouble_scen": {
        "lhs": u64([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123]),

        "rhs": u64([0, 4, 4, 5, 9, 9, 10, 14, 14, 15, 19, 19, 20, 24, 24, 25, 29, 29, 30, 34, 34, 35, 39, 39, 40, 44, 44, 45, 49, 49, 50, 54, 54, 55, 59, 59, 60, 64, 64, 65, 69, 69, 70, 74, 74, 75, 79, 79, 80, 84, 84, 85, 89, 89, 90, 94, 94, 95, 99, 99, 100, 104, 104, 105, 109, 109, 110, 114, 114, 115, 119, 119, 120, 124, 124]),
        "mask": None,
        "expected": u64([5, 9, 15, 19, 25, 29, 35, 39, 45, 49, 55, 59, 65, 69, 75, 79, 85, 89, 95, 99, 105, 109, 115, 119])
    },
    "many_zeros": {
        "lhs": u64([0, 0, 1]),

        "rhs": u64([0, 0, 0, 0, 1]),
        "mask": None,
        "expected": u64([0, 1])
    }
}


@w_scenarios(intersect_scenarios)
def test_int_w_adj_intersects_strided(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs = lhs[::2]
    rhs = rhs[::2]
    expected = np.intersect1d(lhs, rhs)
    lhs_idx, rhs_idx, _, _ = intersect_with_adjacents(lhs, rhs, mask=mask)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


@w_scenarios(intersect_scenarios)
def test_intersect_strided(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs = lhs[::2]
    rhs = rhs[::2]
    expected = np.intersect1d(lhs, rhs)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask=mask)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


@w_scenarios(intersect_scenarios)
def test_intersect(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask=mask)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


@w_scenarios(intersect_scenarios)
def test_int_w_adj_intersects(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx, _, _ = intersect_with_adjacents(lhs, rhs, mask=mask)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


@w_scenarios(intersect_scenarios)
def test_intersect_keep_both(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask=mask, drop_duplicates=False)
    expected_lhs = np.argwhere(np.in1d(lhs, rhs)).flatten()
    expected_rhs = np.argwhere(np.in1d(rhs, lhs)).flatten()
    assert np.all(lhs_idx == expected_lhs)
    assert np.all(rhs_idx == expected_rhs)


@w_scenarios(intersect_scenarios)
def test_intersect_keep_both_strided(lhs, rhs, mask, expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lhs = lhs[::2]
    rhs = rhs[::2]
    expected = np.intersect1d(lhs, rhs)
    lhs_idx, rhs_idx = intersect(lhs, rhs, mask=mask, drop_duplicates=False)
    result = lhs[lhs_idx] & mask
    assert np.all(result == expected)


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
        snp.intersect(rand_arr_1 >> 16, rand_arr_2 >> 16, indices=True, duplicates=snp.DROP)

    def with_snp_ops():
        intersect(rand_arr_1, rand_arr_2, mask)

    def baseline():
        count_odds(rand_arr_1, rand_arr_2)

    def intersect_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            baseline()

    baseline()
    profiler.run(intersect_many)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_masked_intersect_sparse_sparse(benchmark):
    profiler = Profiler(benchmark)

    rand_arr_1 = np.random.randint(0, 500000, 100, dtype=np.uint64)
    rand_arr_2 = np.random.randint(0, 50000, 200, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF00000000)
    rand_arr_1.sort()
    rand_arr_2.sort()

    def with_snp():
        snp.intersect(rand_arr_1 >> 16, rand_arr_2 >> 16, indices=True, duplicates=snp.DROP)

    def with_snp_ops():
        intersect(rand_arr_1, rand_arr_2, mask)

    def baseline():
        count_odds(rand_arr_1, rand_arr_2)

    def intersect_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            baseline()

    baseline()
    profiler.run(intersect_many)


def test_profile_masked_intersect_diff_ranges(benchmark):
    profiler = Profiler(benchmark)

    rand_arr_1 = np.random.randint(0, 50000, 1000000, dtype=np.uint64)
    rand_arr_2 = np.random.randint(0, 500000, 100000, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF00000000)
    rand_arr_1.sort()
    rand_arr_2.sort()

    def with_snp():
        snp.intersect(rand_arr_1 >> 16, rand_arr_2 >> 16, indices=True, duplicates=snp.DROP)

    def with_snp_ops():
        intersect(rand_arr_1, rand_arr_2, mask)

    def baseline():
        count_odds(rand_arr_1, rand_arr_2)

    def intersect_many():
        for _ in range(10):
            baseline()
            with_snp_ops()
            with_snp()

    baseline()
    profiler.run(intersect_many)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("suffix", [128, 185, 24179, 27685, 44358, 45907, 90596])
def test_profile_masked_saved(suffix, benchmark):
    profiler = Profiler(benchmark)

    print(f"Running with {suffix}")
    lhs = np.load(f"fixtures/lhs_{suffix}.npy")
    rhs = np.load(f"fixtures/rhs_{suffix}.npy")
    mask = np.load(f"fixtures/mask_{suffix}.npy")
    print(lhs.shape, rhs.shape)

    def with_snp_ops():
        intersect(lhs, rhs, mask)

    def with_snp():
        snp.intersect(lhs >> 18, rhs >> 18, indices=True, duplicates=snp.DROP)

    def baseline():
        count_odds(lhs, rhs)

    def intersect_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            baseline()

    profiler.run(intersect_many)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_profile_masked_intersect_sparse_dense(benchmark):
    profiler = Profiler(benchmark)

    rand_arr_1 = np.random.randint(0, 500000, 100, dtype=np.uint64)
    rand_arr_2 = np.random.randint(0, 50000, 1000000, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF00000000)
    rand_arr_1.sort()
    rand_arr_2.sort()

    # def with_np():
    #     np.intersect1d(rand_arr_1 >> 16, rand_arr_2 >> 16)  # Not quite same

    def with_snp():
        snp.intersect(rand_arr_1 >> 16, rand_arr_2 >> 16, indices=True, duplicates=snp.DROP)

    def with_snp_ops():
        intersect(rand_arr_1, rand_arr_2, mask)

    def baseline():
        count_odds(rand_arr_1, rand_arr_2)

    def intersect_many():
        for _ in range(10):
            with_snp_ops()
            with_snp()
            baseline()

    baseline()
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
        "delta": 1,
        "lhs_idx_expected": u64arr([0, 2, 4, 6, 8]),
        "rhs_idx_expected": u64arr([0, 1, 2, 3, 4])
    },
    "start_0s": {
        "lhs": u64([0, 0, 3]),
        "rhs": u64([0, 0, 1, 8, 10]),
        "mask": None,
        "delta": 1,
        "lhs_idx_expected": u64arr([0]),
        "rhs_idx_expected": u64arr([2])
    },
    "start_0s_2": {
        "lhs": u64([0, 2, 3]),
        "rhs": u64([0, 1, 2, 8, 10]),
        "mask": None,
        "delta": 1,
        "lhs_idx_expected": u64arr([0]),
        "rhs_idx_expected": u64arr([1]),
    },
    "base_masked": {
        "lhs": u64([0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF]),
        "rhs": u64([0x2F, 0x4F, 0x6F, 0x8F, 0xAF]),
        "mask": 0xF0,
        "delta": 1,
        "lhs_idx_expected": u64arr([0, 2, 4, 6, 8]),
        "rhs_idx_expected": u64arr([0, 1, 2, 3, 4]),
    },
    "base_masked_delta_neg1": {
        "lhs": u64([0x2F, 0x4F, 0x6F, 0x8F, 0xAF]),
        "rhs": u64([0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF]),
        "mask": 0xF0,
        "delta": -1,
        "lhs_idx_expected": u64arr([0, 1, 2, 3, 4]),
        "rhs_idx_expected": u64arr([0, 2, 4, 6, 8])
    },
    "rhs_0": {
        "lhs": u64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "rhs": u64([0, 3]),
        "mask": None,
        "delta": 1,
        "lhs_idx_expected": u64arr([1]),
        "rhs_idx_expected": u64arr([1]),
    },
    "many_adjacent": {
        "lhs": u64([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        "rhs": u64([2, 3]),
        "mask": None,
        "delta": 1,
        "lhs_idx_expected": u64arr([0, 5]),  # As we drop dups
        "rhs_idx_expected": u64arr([0, 1]),
    },
    "trouble_scen": {
        "lhs": u64([1, 274877906945, 549755813889, 824633720833]),
        "rhs": u64([6, 137438953474, 274877906950, 412316860418]),
        "mask": 0xfffffffffffc0000,
        "delta": 1,
        "lhs_expected": u64arr([]),
        "rhs_expected": u64arr([]),
    }
}


@w_scenarios(adj_scenarios)
def test_adjacent(lhs, rhs, mask, delta, lhs_idx_expected, rhs_idx_expected):
    if mask is None:
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    if delta == -1:
        rhs_idx, lhs_idx = adjacent(rhs, lhs, mask)
    else:
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
