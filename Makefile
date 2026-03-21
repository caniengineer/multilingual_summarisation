.PHONY: help install install-dev dev run test test-v lint format fix check docker-build docker-run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install all dependencies including dev
	uv sync --extra dev

dev: ## Start dev server with hot reload
	uv run uvicorn app.main:create_app --factory --reload

run: ## Start production-like server
	uv run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000

test: ## Run test suite
	uv run pytest

test-v: ## Run tests with verbose output
	uv run pytest -v

lint: ## Lint code with ruff
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

fix: ## Auto-fix lint issues
	uv run ruff check --fix .

check: lint test ## Run lint then tests

docker-build: ## Build Docker image
	docker build -t multilingual-summarizer .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env multilingual-summarizer

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
