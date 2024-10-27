
venv:
	@echo "Checking virtualenv..."
	if [ ! -d "venv" ]; then \
		echo "Creating virtualenv..."; \
		python3 -m venv venv; \
		. venv/bin/activate; \
		pip install --upgrade pip; \
		pip install -r requirements.txt; \
	fi


deps: venv
	@if [[ $$VIRTUAL_ENV != "" ]]; then \
		echo "Virtualenv already installed and activated."; \
	else \
		. venv/bin/activate; \
	fi
	# If OSX install open-mpi, llvm, libomp
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		echo "Installing OSX dependencies..."; \
	fi


clean: deps
	@echo "Cleaning..."
	rm -rf build dist
	@echo "Clean any .so files..."
	# Force dlclose
	find . -name "searcharray/*.so" -type f -delete
	find . -name "searcharray/*.c" -type f -delete
	@echo "Cleaning extensions..."
	rm -rf searcharray/*.so
	rm -rf searcharray/*.c
	rm -rf searcharray/bm25/*.so
	rm -rf searcharray/bm25/*.c
	rm -rf searcharray/bm25/*.o
	rm -rf searcharray/phrase/*.o
	rm -rf searcharray/phrase/*.so
	rm -rf searcharray/phrase/*.c
	rm -rf searcharray/utils/*.so
	rm -rf searcharray/utils/*.o
	rm -rf searcharray/utils/*.c
	rm -rf searcharray/roaringish/*.o
	rm -rf searcharray/roaringish/*.so
	rm -rf searcharray/roaringish/*.c
	rm -rf .benchmarks/*.prof
	python setup.py clean --all



destroy: clean
	@echo "Clean deps..."
	deactivate
	rm -rf venv


extensions: clean
	python setup.py build_ext --inplace


annotate:
	@cython -a "$(FILE)"
	open "$(FILE:.pyx=.html)"


test: deps extensions
	@echo "Running tests..."
	python -m pytest --benchmark-skip test


lint: deps
	@echo "Linting..."
	python -m flake8 --max-line-length=120 --ignore=E203,W503,E501,E722,E731,W605 --exclude=venv,build,dist,docs,*.egg-info,*.egg,*.pyc,*.pyo,*.git,__pycache__,.pytest_cache,.benchmarks
	mypy searcharray test
	cython-lint searcharray

fix: deps
	@echo "Fixing..."
	./scripts/fixup.sh searcharray/roaringish/*.pyx


benchmark_dry_run: deps
	python -m pytest -x --benchmark-only


benchmark: deps extensions
	# There is some interaction between doing MSMarco and TMDB benchmarks together 
	# (maybe due to memory usage) that causes TMDB to show as slower. So we run them separately.
	python -m pytest -x --benchmark-only --benchmark-autosave --benchmark-histogram=./.benchmarks/histogram_snp_ops test/test_snp_ops.py
	open ./.benchmarks/histogram_snp_ops.svg
	python -m pytest -x --benchmark-only --benchmark-autosave --benchmark-histogram=./.benchmarks/histogram_tmdb test/test_tmdb.py
	open ./.benchmarks/histogram_tmdb.svg
	python -m pytest -x --benchmark-only --benchmark-autosave --benchmark-histogram=./.benchmarks/histogram_msmarco test/test_msmarco.py
	open ./.benchmarks/histogram_msmarco.svg

benchmark_one: deps extensions
	python -m pytest -x --benchmark-only --benchmark-autosave --benchmark-histogram=./.benchmarks/histogram_one "$(TEST)"
	open ./.benchmarks/histogram_one.svg

benchmark_graph: deps
	python scripts/graph_benchmarks.py "$(TEST)"

benchmark_graph_all: deps
	python scripts/graph_benchmarks.py "$(TEST)" scour


benchmark_graph_clean: deps
	python scripts/graph_benchmarks.py "$(TEST)" clean


favorite_graphs: deps
	python scripts/graph_benchmarks.py "test_msmarco1m_or_search_unwarmed[what is the purpose of]" "$(CLEAN)"
	python scripts/graph_benchmarks.py "test_msmarco1m_or_search_warmed[what is the purpose of]" "$(CLEAN)"
	python scripts/graph_benchmarks.py 'test_msmarco1m_phrase[what is the purpose of]' "$(CLEAN)"
	python scripts/graph_benchmarks.py 'test_msmarco10k_indexing' "$(CLEAN)"


profile: extensions
	./scripts/run_profile.sh $(TEST)

memory_profile:
	rm .benchmarks/memray.bin
	python -m memray run -o .benchmarks/memray.bin test/memray.py
	rm .benchmarks/memray-flamegraph-memray.html
	python -m memray flamegraph .benchmarks/memray.bin
	open .benchmarks/memray-flamegraph-memray.html


build: deps test lint
	@echo "Building..."
	python3 -m build --sdist
	python3 -m build --wheel


twine:
	@echo "Installing twine..."
	pip install twine


publish: twine
	@echo "Publishing..."
	python scripts/publish.py


help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  deps            Install dependencies"
	@echo "  test            Run tests"
	@echo "  build           Build package (local)"
	@echo "  clean           Clean build files"
	@echo "  destroy         Completely destroy the dev env"
	@echo "  help            Show this help message"
	@echo "  publish         Fetch current sha wheels github and publish to pypi" 

.DEFAULT_GOAL := build
