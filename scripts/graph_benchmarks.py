import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass


def all_benchmark_dirs(subdir='Darwin-CPython-3.12-64bit/'):
    return [os.path.join(f, subdir)
            for f in os.listdir('.') if f.startswith('.benchmarks')]


def git_shas():
    os.system('git log --pretty=format:"%H" > git_shas.txt')
    with open('git_shas.txt', 'r') as f:
        shas = f.readlines()
        return [sha.strip() for sha in shas]
    return None


def all_benchmarks_files(benchmark_name,
                         benchmark_dirs=[".benchmarks/"]):

    files = []
    for directory in benchmark_dirs:
        this_dir_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
        files.extend(this_dir_files)
    return files


@dataclass
class Benchmark:
    name: str
    date: str
    commit_time: datetime
    commit_sha: str
    dirty: bool
    std_dev: float
    mean_time: float
    prefix: str


def renamed(name):
    # Handle renamed benchmarks mapping old names to the new ones
    if 'test_msmarco[' in name:
        return name.replace('test_msmarco[', 'test_msmarco100k_phrase[')
    if 'test_msmarco100k[' in name:
        return name.replace('test_msmarco100k[', 'test_msmarco100k_phrase[')
    if name.startswith('test_msmarco1m['):
        return name.replace('test_msmarco1m[', 'test_msmarco1m_phrase[')
    return name


def json_to_benchmark(json_row, filename, full_file) -> Benchmark:
    prefix = filename.split('/')[-1][:4]
    name = renamed(json_row['name'])
    return Benchmark(
        name=name,
        date=full_file['datetime'],
        commit_sha=full_file['commit_info']['id'],
        dirty=full_file['commit_info']['dirty'],
        std_dev=json_row['stats']['stddev'],
        mean_time=json_row['stats']['mean'],
        commit_time=datetime.fromisoformat(full_file['commit_info']['time']),
        prefix=prefix
    )


def show_plot(name, commits, times, std_devs):
    # Extract dates and times
    # Plotting
    plt.plot(commits, times, marker='o', label='Mean Execution Time')
    plt.fill_between(commits, [m - s for m, s in zip(times, std_devs)], [m + s for m, s in zip(times, std_devs)], color='gray', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Commit Sha')
    plt.ylabel('Mean Execution Time (secs)')
    plt.title(f'Benchmark Over Time: {name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Annotate x, y with times
    last_time = 0xFFFFFFFFFFFFFFFF
    for i, (commit, time) in enumerate(zip(commits, times)):
        # If a big change, then onnotate
        prop_change = abs(time - last_time) / last_time
        if prop_change > 0.20 or i == 0 or i == len(commits) - 1:
            last_time = time
            y_offset = (i % 2) * 2 - 1
            if i == 0:
                y_offset = 1
            if i == len(commits) - 1:
                y_offset = -1
            plt.annotate(f'{time:.3f}', (commit, time),
                         textcoords="offset points", xytext=(0, 10 * y_offset), ha='center')
    plt.show()


def is_buggy_sha(sha):
    """Benchmarks ran but with big bugs and should be ignored.

    Some of these are weird runs with a buggy tag, or from a debugging session.
    """
    banned_shas = [
        '330ba9a9',
        '0b6d94c3',
    ]
    return sha[:8] in banned_shas


def graph_benchmark(benchmark_name,
                    only_clean=False,
                    directories='.benchmarks/Darwin-CPython-3.12-64bit/'):

    if isinstance(directories, str):
        directories = [directories]

    files = all_benchmarks_files(benchmark_name, directories)
    # Data structure to hold the extracted data
    benchmarks = {}

    legit_shas = git_shas()

    # Extract data
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for bench in data['benchmarks']:
                benchmark = json_to_benchmark(bench, file, data)

                if benchmark.commit_sha in legit_shas and not is_buggy_sha(benchmark.commit_sha):
                    if benchmark.name not in benchmarks:
                        benchmarks[benchmark.name] = []
                    if not only_clean or (only_clean and not benchmark.dirty):
                        benchmarks[benchmark.name].append(benchmark)

    benchmarks[benchmark_name].sort(key=lambda x: (x.commit_time, x.commit_sha))
    show_plot(benchmark_name,
              [b.prefix + "_" + b.commit_sha[:8] for b in benchmarks[benchmark_name]],
              [b.mean_time for b in benchmarks[benchmark_name]],
              [b.std_dev for b in benchmarks[benchmark_name]])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        graph_benchmark(sys.argv[1])
    elif len(sys.argv) == 3:
        if sys.argv[2] == 'clean':
            graph_benchmark(sys.argv[1], True)
        elif sys.argv[2] == 'scour':
            graph_benchmark(sys.argv[1], only_clean=False,
                            directories=all_benchmark_dirs())
    else:
        print('Usage: python graph_benchmarks.py benchmark_name [benchmark_dir]')
