.PHONY: help install dev lint format check test test-cov type clean pre-commit ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	uv sync

dev: ## Install project with dev dependencies
	uv sync --group dev

lint: ## Run ruff linter
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check: lint type test ## Run all checks (lint + type + tests)

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=term-missing

type: ## Run type checker
	uv run ty check src/

clean: ## Remove build artifacts and caches
	rm -rf .ruff_cache .pytest_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

pre-commit: ## Install pre-commit hooks
	uv run pre-commit install

ci: check test ## Run full CI pipeline locally
