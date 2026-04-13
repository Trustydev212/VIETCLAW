.PHONY: help install install-dev test lint validate chunk index export clean

help: ## Hien thi huong dan
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Cai dat package (production)
	pip install -e .

install-dev: ## Cai dat package voi dev dependencies
	pip install -e ".[all]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Chay tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Chay tests voi coverage report
	python -m pytest tests/ -v --tb=short --cov=vietlaw --cov=scripts --cov-report=term-missing

lint: ## Kiem tra code style (ruff)
	python -m ruff check vietlaw/ scripts/ tests/

lint-fix: ## Tu dong sua loi lint
	python -m ruff check --fix vietlaw/ scripts/ tests/

# ---------------------------------------------------------------------------
# Data processing pipeline
# ---------------------------------------------------------------------------

validate: ## Kiem tra metadata cua tat ca van ban
	python scripts/validate_metadata.py --data-dir data/ --schema config/metadata_schema.json

chunk: ## Chunk van ban phap luat
	python scripts/chunk_documents.py --input data/ --output chunks/ --config config/rag_config.yaml

index: ## Tao ChromaDB vector index (can OPENAI_API_KEY)
	python scripts/build_index.py --chunks-dir chunks/ --config config/rag_config.yaml

export: ## Xuat embeddings ra file
	python scripts/export_embeddings.py --collection vietlaw_laws --format parquet --output exports/embeddings

pipeline: validate chunk index ## Chay toan bo pipeline: validate -> chunk -> index

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Xoa cac file tam
	rm -rf chunks/ index/ exports/ vectordb/ logs/
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
