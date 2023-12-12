import pytest
from typing import Dict, Any
import cProfile
import sys


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


class JustBenchmarkProfiler:
    def __init__(self, benchmark):
        self.benchmark = benchmark

    def run(self, func, *args, **kwargs):
        rval = self.benchmark(func, *args, **kwargs)
        return rval


class CProfileProfiler:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.name = benchmark.name
        self.name = self.name.split('/')[-1].replace(':', '_').replace('.', '_').replace('::', '_').replace('[', '_').replace(']', '_').replace(' ', '_')
        self.cprofiler = cProfile.Profile()

    def run(self, func, *args, **kwargs):
        rval = self.cprofiler.runcall(func, *args, **kwargs)
        self.cprofiler.dump_stats(f".benchmarks/{self.name}.prof")
        return rval


if '--benchmark-disable' in sys.argv:
    Profiler = CProfileProfiler
else:
    Profiler = JustBenchmarkProfiler
