# SearchArray

A pandas extension array for implementing full text search with BM25/TFIDF scoring.

## Setup

This project uses a virtualenv. Activate it before running any commands:

```bash
source venv/bin/activate
```

## Common Commands

```bash
# Build Cython extensions (required before running tests)
make extensions

# Run tests (builds extensions first)
make test

# Run linting (flake8, mypy, cython-lint)
make lint

# Run benchmarks
make benchmark

# Clean build artifacts
make clean
```

## Performance

This is a performance-sensitive codebase. Keep the following in mind:

- The core indexing and scoring operations must be fast
- Cython extensions (`.pyx` files) are used for hot paths
- Run benchmarks (`make benchmark`) when making changes to core algorithms
- Profile with `make profile TEST=<test_name>` to identify bottlenecks
- Memory efficiency matters - target 100K-1M docs for offline use

## Project Structure

- `searcharray/` - Main package with Cython extensions
- `test/` - Tests and benchmarks (pytest-benchmark)
- `fixtures/` - Test data files
- `scripts/` - Build and profiling utilities
