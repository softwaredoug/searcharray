import pytest
import datetime
import pandas as pd
from typing import Dict, Any, cast, Sequence, Type, Union
import cProfile
import sys


def data_dir():
    return 'data/'


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


class JustBenchmarkProfiler:
    def __init__(self, benchmark, warmup=True):
        self.benchmark = benchmark

    def run(self, func, *args, **kwargs):
        rval = self.benchmark(func, *args, **kwargs)
        return rval


class CProfileProfiler:
    def __init__(self, benchmark, warmup=True):
        self.benchmark = benchmark
        self.warmup = warmup

        def replace_all(text, replace_with='_'):
            for char in ['/', ':', '.', '::', '[', ']', ' ']:
                text = text.replace(char, replace_with)
            return text

        self.name = benchmark.name
        self.name = replace_all(self.name.split('/')[-1])
        self.cprofiler = cProfile.Profile()

    def run(self, func, *args, **kwargs):
        if self.warmup:
            func(*args, **kwargs)
        rval = self.cprofiler.runcall(func, *args, **kwargs)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.cprofiler.dump_stats(f".benchmarks/{self.name}_{timestamp}.prof")
        return rval


def naive_find_term(text: pd.Series, term: str,
                    tokenizer):
    text_as_tokens = text.apply(tokenizer)
    return text_as_tokens.apply(lambda tokens: term in tokens)


Profiler: Union[Type[JustBenchmarkProfiler], Type[CProfileProfiler]]

if '--benchmark-disable' in sys.argv:
    Profiler = CProfileProfiler
else:
    Profiler = JustBenchmarkProfiler


profile_enabled = '--benchmark-only' in sys.argv or '--benchmark-disable' in sys.argv
stress_test_enabled = '--searcharray-stress-test' in sys.argv
