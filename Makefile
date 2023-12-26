
venv:
	@echo "Checking virtualenv..."
	@if [ ! -d "venv" ]; then \
		echo "Creating virtualenv..."; \
		python3 -m venv venv; \
		. venv/bin/activate; \
		pip install --upgrade pip; \
		pip install -r requirements.txt
	fi


deps: venv
	@if [[ $$VIRTUAL_ENV != "" ]]; then \
		echo "Virtualenv already installed and activated."; \
	else \
		. venv/bin/activate; \
	fi


clean:
	@echo "Cleaning..."
	rm -rf dist
	@echo "Clean deps..."
	deactivate
	rm -rf venv


test: deps
	@echo "Running tests..."
	python -m pytest --benchmark-skip test


lint: deps
	@echo "Linting..."
	python -m flake8 --max-line-length=120 --ignore=E203,W503,E501,E722,E731,W605 --exclude=venv,build,dist,docs,*.egg-info,*.egg,*.pyc,*.pyo,*.git,__pycache__,.pytest_cache,.benchmarks
	mypy searcharray tests


benchmark_dry_run: deps
	python -m pytest -x --benchmark-only


benchmark: deps
	python -m pytest -x --benchmark-only --benchmark-autosave --benchmark-histogram=./.benchmarks/histogram
	open ./.benchmarks/histogram.svg

benchmark_graph: deps
	python scripts/graph_benchmarks.py "$(TEST)"

profile:
	python -m pytest -x --benchmark-disable "$(TEST)"
	snakeviz ./.benchmarks/last.prof


build: deps test
	@echo "Building..."
	python3 -m build --sdist
	python3 -m build --wheel


twine:
	@echo "Installing twine..."
	pip install twine


publish: twine
	@echo "Publishing..."
	twine upload --skip-existing dist/*


help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  deps            Install dependencies"
	@echo "  test            Run tests"
	@echo "  build           Build package"
	@echo "  clean           Clean build files"
	@echo "  help            Show this help message"
	@echo "  publish         Publish package to PyPI"

.DEFAULT_GOAL := build
