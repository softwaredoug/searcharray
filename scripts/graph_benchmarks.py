import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt


def graph_benchmark(benchmark_name,
                    only_clean=False,
                    benchmark_dir='.benchmarks/Darwin-CPython-3.11-64bit/'):

    # Collect all benchmark files
    files = [os.path.join(benchmark_dir, f) for f in os.listdir(benchmark_dir) if f.endswith('.json')]

    # Data structure to hold the extracted data
    benchmarks = {}

    # Extract data
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for bench in data['benchmarks']:
                name = bench['name']  # Benchmark name
                date = data['datetime']  # Date of the benchmark
                mean_time = bench['stats']['mean']  # Mean execution time
                std_dev = bench['stats']['stddev']  # Standard deviation
                commit_sha = data['commit_info']['id']  # Commit SHA
                commit_time = datetime.fromisoformat(data['commit_info']['time'])  # Commit time
                dirty = data['commit_info']['dirty']  # Dirty flag
                prefix = file.split('/')[-1][:4]

                if name not in benchmarks:
                    benchmarks[name] = []
                if not only_clean or (only_clean and not dirty):
                    benchmarks[name].append((date, commit_time, prefix, commit_sha, dirty, std_dev, mean_time))

    benchmarks[benchmark_name].sort(key=lambda x: (x[1], x[2]))

    # Extract dates and times
    commits = [x[2] + "_" + x[3][:8] for x in benchmarks[benchmark_name]]
    times = [x[-1] for x in benchmarks[benchmark_name]]
    std_devs = [x[-2] for x in benchmarks[benchmark_name]]

    # Plotting
    plt.plot(commits, times, marker='o', label='Mean Execution Time')
    plt.fill_between(commits, [m - s for m, s in zip(times, std_devs)], [m + s for m, s in zip(times, std_devs)], color='gray', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Commit Sha')
    plt.ylabel('Mean Execution Time (secs)')
    plt.title(f'Benchmark Over Time: {benchmark_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        graph_benchmark(sys.argv[1])
    elif len(sys.argv) == 3:
        graph_benchmark(sys.argv[1], sys.argv[2] == 'clean')
    else:
        print('Usage: python graph_benchmarks.py benchmark_name [benchmark_dir]')
