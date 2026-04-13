FROM python:3.12-slim AS base

LABEL maintainer="Trustydev212"
LABEL description="VIETLAW - Kho Du Lieu Phap Luat Viet Nam cho AI/RAG"

WORKDIR /app

# System dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
    pip install --no-cache-dir -r requirements.txt

# Copy project files.
COPY . .

# ---------------------------------------------------------------------------
# Stage: validate - Run metadata validation
# ---------------------------------------------------------------------------
FROM base AS validate
CMD ["python", "scripts/validate_metadata.py", "--data-dir", "data/", "--schema", "config/metadata_schema.json"]

# ---------------------------------------------------------------------------
# Stage: chunk - Run document chunking
# ---------------------------------------------------------------------------
FROM base AS chunk
CMD ["python", "scripts/chunk_documents.py", "--input", "data/", "--output", "chunks/"]

# ---------------------------------------------------------------------------
# Stage: default - Interactive Python shell
# ---------------------------------------------------------------------------
FROM base AS default
CMD ["python", "-c", "from vietlaw import VietlawRAG; print('VIETLAW ready. Use VietlawRAG to start.')"]
