#!/usr/bin/env python3
"""
build_index.py - Build a ChromaDB vector index from pre-chunked JSON documents.

Loads JSON chunk files produced by chunk_documents.py, generates embeddings
via the OpenAI text-embedding-3-large model (or the configured fallback), and
persists them in a ChromaDB collection with full metadata for filtered retrieval.

Usage:
    python scripts/build_index.py \
        --chunks-dir chunks/ \
        --config     config/rag_config.yaml \
        --collection vietclaw_laws
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_index")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML configuration file; return empty dict on failure."""
    if not config_path.exists():
        logger.warning("Config not found at %s; using defaults.", config_path)
        return {}
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def load_chunks_from_dir(chunks_dir: Path) -> list[dict[str, Any]]:
    """
    Recursively load every ``*.json`` file under *chunks_dir*.

    Each JSON file is expected to be a list of chunk dicts as produced by
    ``chunk_documents.py``.
    """
    json_files = sorted(chunks_dir.rglob("*.json"))
    if not json_files:
        logger.error("No JSON chunk files found under '%s'.", chunks_dir)
        sys.exit(1)

    all_chunks: list[dict[str, Any]] = []
    for json_path in json_files:
        try:
            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                all_chunks.extend(data)
            else:
                logger.warning("Unexpected format in '%s'; skipping.", json_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load '%s': %s", json_path, exc)

    logger.info("Loaded %d chunks from %d files.", len(all_chunks), len(json_files))
    return all_chunks


# ---------------------------------------------------------------------------
# Metadata sanitisation for ChromaDB
# ---------------------------------------------------------------------------

_ALLOWED_CHROMA_TYPES = (str, int, float, bool)


def sanitise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """
    ChromaDB requires metadata values to be str, int, float, or bool.

    Lists and other complex values are serialised to JSON strings so they
    can still be stored and later deserialised by the retrieval layer.
    """
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, _ALLOWED_CHROMA_TYPES):
            clean[key] = value
        else:
            # Fallback: JSON-encode lists, dicts, dates, etc.
            try:
                clean[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                clean[key] = str(value)
    return clean


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """
    Thin wrapper around the OpenAI Embeddings API.

    Handles batching and basic retry/back-off so the caller does not need
    to think about rate limits.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install it with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed *texts* in batches, returning a list of float vectors in the
        same order as the input.
        """
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            logger.info(
                "Embedded %d/%d texts.",
                min(batch_start + self.batch_size, len(texts)),
                len(texts),
            )

        return all_embeddings

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Call the API for one batch, retrying on transient errors."""
        import openai  # noqa: PLC0415

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # The API returns embeddings ordered by index.
                ordered = sorted(response.data, key=lambda e: e.index)
                return [e.embedding for e in ordered]
            except openai.RateLimitError:
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %.1fs.",
                    attempt,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
            except openai.APIError as exc:
                logger.error("OpenAI API error: %s", exc)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Failed to embed batch after {self.max_retries} retries."
        )


# ---------------------------------------------------------------------------
# ChromaDB index builder
# ---------------------------------------------------------------------------


class ChromaIndexBuilder:
    """
    Inserts chunk embeddings and metadata into a ChromaDB collection.

    The collection is created (or re-opened) at the persist directory
    specified in the config.  Documents are upserted in batches so a
    previously interrupted run can be resumed safely.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        distance_metric: str = "cosine",
        upsert_batch_size: int = 500,
    ) -> None:
        try:
            import chromadb  # noqa: PLC0415
            from chromadb.config import Settings  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'chromadb' package is required. Install it with: pip install chromadb"
            ) from exc

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        self.upsert_batch_size = upsert_batch_size
        logger.info(
            "ChromaDB collection '%s' opened (documents already stored: %d).",
            collection_name,
            self.collection.count(),
        )

    def upsert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert *chunks* with their *embeddings* into the collection."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        total = len(chunks)
        for batch_start in range(0, total, self.upsert_batch_size):
            batch_chunks = chunks[batch_start : batch_start + self.upsert_batch_size]
            batch_embeddings = embeddings[batch_start : batch_start + self.upsert_batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["content"] for c in batch_chunks]
            metadatas = [
                sanitise_metadata(
                    {
                        **c.get("metadata", {}),
                        "source_file": c.get("source_file", ""),
                        "parent_context": c.get("parent_context", ""),
                    }
                )
                for c in batch_chunks
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Upserted %d/%d chunks.",
                min(batch_start + self.upsert_batch_size, total),
                total,
            )

        logger.info(
            "Index now contains %d documents.", self.collection.count()
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    chunks_dir: Path,
    config_path: Path,
    collection_name: str | None,
) -> None:
    """Full build pipeline: load chunks → embed → upsert into ChromaDB."""
    cfg = load_config(config_path)

    # ---- embedding config ------------------------------------------------
    emb_cfg: dict[str, Any] = cfg.get("embedding", {})
    model: str = emb_cfg.get("model", "text-embedding-3-large")
    batch_size: int = int(emb_cfg.get("batch_size", 100))

    # ---- vector store config ---------------------------------------------
    vs_cfg: dict[str, Any] = cfg.get("vector_store", {})
    chroma_cfg: dict[str, Any] = vs_cfg.get("chroma", {})
    persist_dir: str = chroma_cfg.get("persist_directory", "./vectordb/chroma")
    distance_metric: str = vs_cfg.get("distance_metric", "cosine")
    resolved_collection: str = (
        collection_name
        or vs_cfg.get("collection_name", "vietclaw_laws")
    )

    # ---- load chunks -----------------------------------------------------
    chunks = load_chunks_from_dir(chunks_dir)
    if not chunks:
        logger.error("No chunks to index. Exiting.")
        sys.exit(1)

    # ---- build embedder --------------------------------------------------
    try:
        embedder = OpenAIEmbedder(model=model, batch_size=batch_size)
    except (ImportError, EnvironmentError) as exc:
        logger.error("Cannot initialise embedder: %s", exc)
        sys.exit(1)

    # ---- generate embeddings --------------------------------------------
    texts = [c["content"] for c in chunks]
    logger.info(
        "Generating embeddings for %d chunks using model '%s'…",
        len(texts),
        model,
    )
    try:
        embeddings = embedder.embed(texts)
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding failed: %s", exc)
        sys.exit(1)

    # ---- build ChromaDB index -------------------------------------------
    try:
        builder = ChromaIndexBuilder(
            persist_directory=persist_dir,
            collection_name=resolved_collection,
            distance_metric=distance_metric,
        )
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    builder.upsert(chunks, embeddings)
    logger.info("Index build complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB vector index from chunked Vietnamese legal documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory containing JSON chunk files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rag_config.yaml"),
        help="Path to the RAG configuration YAML file.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help=(
            "ChromaDB collection name. Overrides the value in config "
            "when provided."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    run(
        chunks_dir=args.chunks_dir.resolve(),
        config_path=args.config.resolve(),
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
