
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


test: deps
	@echo "Running tests..."
	python -m pytest test


build: deps
	@echo "Building..."
	python3 -m build --sdist
	python3 -m build --wheel


help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  deps            Install dependencies"
	@echo "  test            Run tests"
	@echo "  build           Build package"
	@echo "  help            Show this help message"
