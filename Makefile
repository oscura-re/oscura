.PHONY: help install dev test lint format typecheck security clean docs

help:
	@echo "Oscura Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install package"
	@echo "  dev         Install with all extras + git hooks"
	@echo "  sync        Sync dependencies with lockfile"
	@echo ""
	@echo "Quality (uses optimized scripts):"
	@echo "  test        Run tests (optimal parallel config)"
	@echo "  test-fast   Run tests without coverage"
	@echo "  test-cov    Run tests with coverage report"
	@echo "  lint        Run linter"
	@echo "  format      Format code"
	@echo "  typecheck   Run type checker"
	@echo "  check       Run all checks (lint, typecheck, test)"
	@echo "  fix         Auto-fix all fixable issues"
	@echo ""
	@echo "Testing:"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-compliance  Run IEEE/JEDEC compliance tests"
	@echo "  test-benchmark   Run performance benchmarks"
	@echo ""
	@echo "Other:"
	@echo "  clean       Remove build artifacts"
	@echo "  docs        Build documentation"
	@echo "  docs-serve  Serve documentation locally"

# =============================================================================
# Setup
# =============================================================================

install:
	uv pip install -e .

dev:
	uv sync --all-extras
	./scripts/setup/install-hooks.sh

sync:
	uv sync --all-extras

# =============================================================================
# Quality Checks (use optimized scripts)
# =============================================================================

test:
	./scripts/test.sh

test-fast:
	./scripts/test.sh --fast

test-cov:
	./scripts/testing/run_coverage.sh

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-compliance:
	uv run pytest -v -m compliance

test-benchmark:
	uv run pytest -v -m benchmark --benchmark-only

lint:
	./scripts/quality/lint.sh

format:
	./scripts/quality/format.sh

typecheck:
	uv run mypy src/

check:
	./scripts/check.sh

fix:
	./scripts/fix.sh

security:
	uv run bandit -c pyproject.toml -r src/
	uv run safety check || true

# =============================================================================
# Documentation
# =============================================================================

docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

# =============================================================================
# Cleanup
# =============================================================================

clean:
	./scripts/clean.sh

# =============================================================================
# Release
# =============================================================================

build:
	uv build

publish-test:
	uv publish --repository testpypi

publish:
	uv publish
