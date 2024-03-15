import numpy as np
import pytest
from searcharray.utils.bitcount import bit_count64
from searcharray.roaringish import popcount64
from test_utils import w_scenarios
from test_utils import Profiler, profile_enabled


scenarios = {
    "base": {
        "bits": np.asarray([np.uint64(0b0001)]),
        "expected": [1],
    },
    "alternating_ones_lower32": {
        "bits": np.asarray([np.uint64(0b01010101010101010101010101010101)]),
        "expected": [16],
    },
    "alternating_ones_upper32": {
        "bits": np.asarray([np.uint64(0b0101010101010101010101010101010100000000000000000000000000000000)]),
        "expected": [16],
    },
    "one_at_end": {
        "bits": np.asarray([np.uint64(0b0101010101010101010101010101010100000000000000000000000000000001)]),
        "expected": [17],
    },
    "alt_ones": {
        "bits": np.asarray([np.uint64(0b0101010101010101010101010101010101010101010101010101010101010101)]),
        "expected": [32],
    },
    "all_ones": {
        "bits": np.asarray([np.uint64(0b1111111111111111111111111111111111111111111111111111111111111111)]),
        "expected": [64],
    },
}


@w_scenarios(scenarios)
def test_bitcount64(bits, expected):
    bits_before = np.copy(bits)
    assert bit_count64(bits) == expected
    assert np.array_equal(bits, bits_before)


@w_scenarios(scenarios)
def test_popocount64(bits, expected):
    bits_before = np.copy(bits)
    popcont = popcount64(bits)
    assert np.all(popcont == expected)
    assert np.array_equal(bits, bits_before)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_bitcount64_benchmark(benchmark):
    profiler = Profiler(benchmark)

    def bitcounts(arr, times=100):
        for _ in range(times):
            bit_count64(arr)

    def popcounts(arr, times=100):
        for _ in range(times):
            popcount64(arr)

    def run_bitcounts():
        arr_size = 100000
        np.random.seed(0)
        arr = np.random.randint(0, 2**64, size=arr_size, dtype=np.uint64)
        bitcounts(arr)
        popcounts(arr)

    profiler.run(run_bitcounts)
