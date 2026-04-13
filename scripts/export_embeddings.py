#!/usr/bin/env python3
"""
export_embeddings.py - Export embeddings from a ChromaDB collection.

Retrieves stored embedding vectors and their metadata from ChromaDB and
writes them to one of three interchange formats:

  numpy   - ``.npz`` archive: arrays ``embeddings``, ``ids``, ``documents``,
             plus a companion ``<output>_metadata.json`` for the metadata dicts.
  json    - A single ``.json`` file: list of records, each containing the
             embedding vector, document text, id and metadata.
  parquet - A ``.parquet`` file (requires ``pyarrow`` and ``pandas``):
             one row per chunk; the embedding vector is stored as a
             fixed-size list column.

An optional ``--filter`` argument accepts a JSON string of key/value pairs
that are passed directly to ChromaDB's ``where`` clause for server-side
metadata filtering before export.

Usage:
    # Export everything to numpy
    python scripts/export_embeddings.py \
        --collection vietclaw_laws \
        --format     numpy \
        --output     exports/vietclaw_embeddings

    # Export only "Luat" documents to parquet
    python scripts/export_embeddings.py \
        --collection vietclaw_laws \
        --format     parquet \
        --output     exports/luat_embeddings \
        --filter     '{"loai_van_ban": "Luat"}'

    # Limit to the first 1000 records
    python scripts/export_embeddings.py \
        --collection vietclaw_laws \
        --format     json \
        --output     exports/sample \
        --limit      1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
logger = logging.getLogger("export_embeddings")

# Supported output formats.
SUPPORTED_FORMATS = ("numpy", "json", "parquet")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the optional YAML config; return empty dict on failure."""
    if not config_path.exists():
        logger.debug("Config not found at %s; using defaults.", config_path)
        return {}
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------


def open_collection(
    persist_directory: str,
    collection_name: str,
) -> Any:
    """
    Open an existing ChromaDB persistent collection.

    Raises ``ImportError`` if chromadb is not installed.
    Raises ``ValueError`` if the collection does not exist.
    """
    try:
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'chromadb' package is required. Install it with: pip install chromadb"
        ) from exc

    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise ValueError(
            f"ChromaDB persist directory does not exist: {persist_directory}"
        )

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:  # chromadb raises a generic Exception on missing collection
        raise ValueError(
            f"Collection '{collection_name}' not found in '{persist_directory}'."
        ) from exc

    logger.info(
        "Opened collection '%s' (%d documents).",
        collection_name,
        collection.count(),
    )
    return collection


def fetch_records(
    collection: Any,
    where_filter: dict[str, Any] | None,
    limit: int | None,
    batch_size: int = 5000,
) -> dict[str, list[Any]]:
    """
    Fetch ids, embeddings, documents, and metadatas from *collection*.

    ChromaDB's ``get()`` may time-out or consume excessive memory on very
    large collections, so records are retrieved in batches when a *limit*
    is not set.

    Returns a dict with keys: ``ids``, ``embeddings``, ``documents``,
    ``metadatas``.
    """
    total = collection.count()
    if total == 0:
        logger.warning("Collection is empty.")
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    effective_limit = min(limit, total) if limit else total
    logger.info("Fetching %d/%d records…", effective_limit, total)

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    # Build the base kwargs for collection.get().
    get_kwargs: dict[str, Any] = {
        "include": ["embeddings", "documents", "metadatas"],
    }
    if where_filter:
        get_kwargs["where"] = where_filter

    offset = 0
    while offset < effective_limit:
        current_batch = min(batch_size, effective_limit - offset)
        get_kwargs["limit"] = current_batch
        get_kwargs["offset"] = offset

        result = collection.get(**get_kwargs)

        batch_ids: list[str] = result.get("ids") or []
        if not batch_ids:
            break  # No more records.

        all_ids.extend(batch_ids)
        all_embeddings.extend(result.get("embeddings") or [[] for _ in batch_ids])
        all_documents.extend(result.get("documents") or ["" for _ in batch_ids])
        all_metadatas.extend(result.get("metadatas") or [{} for _ in batch_ids])

        offset += len(batch_ids)
        logger.info("Fetched %d/%d records.", offset, effective_limit)

        if len(batch_ids) < current_batch:
            break  # Reached end of results.

    logger.info("Total fetched: %d records.", len(all_ids))
    return {
        "ids": all_ids,
        "embeddings": all_embeddings,
        "documents": all_documents,
        "metadatas": all_metadatas,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_numpy(records: dict[str, list[Any]], output_base: Path) -> Path:
    """
    Save embeddings as a ``.npz`` archive plus a companion ``_metadata.json``.

    The ``.npz`` contains:
      - ``embeddings``  : float32 array of shape (N, D)
      - ``ids``         : str array of shape (N,)
      - ``documents``   : str array of shape (N,)

    Metadata dicts are written separately because numpy cannot store
    heterogeneous dicts inside an npz archive.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'numpy' package is required. Install it with: pip install numpy"
        ) from exc

    embeddings_array = np.array(records["embeddings"], dtype=np.float32)
    ids_array = np.array(records["ids"], dtype=object)
    documents_array = np.array(records["documents"], dtype=object)

    npz_path = output_base.with_suffix(".npz")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(npz_path),
        embeddings=embeddings_array,
        ids=ids_array,
        documents=documents_array,
    )
    logger.info("Saved embeddings array to %s  (shape: %s)", npz_path, embeddings_array.shape)

    meta_path = output_base.parent / (output_base.stem + "_metadata.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"ids": records["ids"], "metadatas": records["metadatas"]},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved metadata companion to %s", meta_path)

    return npz_path


def export_json(records: dict[str, list[Any]], output_base: Path) -> Path:
    """
    Save all data (embeddings + metadata + documents) as a single JSON file.

    Each element of the output list is::

        {
            "id":        "<chunk_id>",
            "document":  "<text content>",
            "embedding": [0.123, ...],
            "metadata":  { ... }
        }
    """
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")

    records_list = [
        {
            "id": id_,
            "document": doc,
            "embedding": emb,
            "metadata": meta,
        }
        for id_, doc, emb, meta in zip(
            records["ids"],
            records["documents"],
            records["embeddings"],
            records["metadatas"],
        )
    ]

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records_list, fh, ensure_ascii=False, indent=2)

    logger.info("Saved %d records to %s", len(records_list), json_path)
    return json_path


def export_parquet(records: dict[str, list[Any]], output_base: Path) -> Path:
    """
    Save embeddings and metadata as a Parquet file.

    Requires ``pandas`` and ``pyarrow``.  The embedding vector is stored as
    a ``list<float32>`` column named ``embedding``.
    """
    try:
        import pandas as pd  # noqa: PLC0415
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'pandas' and 'pyarrow' packages are required for parquet export. "
            "Install them with: pip install pandas pyarrow"
        ) from exc

    output_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_base.with_suffix(".parquet")

    # Build a flat DataFrame from metadata dicts first.
    df = pd.DataFrame(records["metadatas"])

    # Add the fixed columns.
    df.insert(0, "id", records["ids"])
    df.insert(1, "document", records["documents"])

    # Embeddings need to be stored as a fixed-size list in Arrow.
    embeddings = records["embeddings"]
    if embeddings and embeddings[0]:
        dim = len(embeddings[0])
        emb_type = pa.list_(pa.float32(), dim)
    else:
        emb_type = pa.list_(pa.float32())

    # Convert the pandas DataFrame to an Arrow table, then add embedding column.
    table = pa.Table.from_pandas(df, preserve_index=False)
    emb_array = pa.array(embeddings, type=emb_type)
    table = table.append_column(
        pa.field("embedding", emb_type), emb_array
    )

    pq.write_table(table, str(parquet_path), compression="snappy")
    logger.info(
        "Saved %d rows to %s  (columns: %s)",
        len(records["ids"]),
        parquet_path,
        table.column_names,
    )
    return parquet_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    collection_name: str,
    output_base: Path,
    fmt: str,
    where_filter: dict[str, Any] | None,
    limit: int | None,
    config_path: Path,
) -> None:
    """Full export pipeline."""
    cfg = load_config(config_path)
    vs_cfg: dict[str, Any] = cfg.get("vector_store", {})
    chroma_cfg: dict[str, Any] = vs_cfg.get("chroma", {})
    persist_dir: str = chroma_cfg.get("persist_directory", "./vectordb/chroma")

    if fmt not in SUPPORTED_FORMATS:
        logger.error(
            "Unsupported format '%s'. Choose from: %s",
            fmt,
            ", ".join(SUPPORTED_FORMATS),
        )
        sys.exit(1)

    # Open ChromaDB collection.
    try:
        collection = open_collection(persist_dir, collection_name)
    except (ImportError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Fetch records.
    records = fetch_records(collection, where_filter, limit)
    if not records["ids"]:
        logger.warning("No records matched the filter criteria. Nothing exported.")
        sys.exit(0)

    # Export.
    dispatch = {
        "numpy":   export_numpy,
        "json":    export_json,
        "parquet": export_parquet,
    }
    try:
        output_file = dispatch[fmt](records, output_base)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Export complete → %s", output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_filter(raw: str) -> dict[str, Any]:
    """
    Parse the ``--filter`` JSON string into a dict.

    Accepts either a simple equality map like ``{"loai_van_ban": "Luat"}``
    or a full ChromaDB ``where`` expression.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--filter must be valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--filter must be a JSON object.")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export embeddings from a ChromaDB collection to numpy / json / parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="vietclaw_laws",
        help="Name of the ChromaDB collection to export.",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default="numpy",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/embeddings"),
        help=(
            "Output path *without* extension.  The appropriate extension "
            "(.npz, .json, or .parquet) is appended automatically."
        ),
    )
    parser.add_argument(
        "--filter",
        type=parse_filter,
        default=None,
        metavar="JSON",
        help=(
            "Metadata filter as a JSON object (ChromaDB 'where' clause). "
            'Example: \'{"loai_van_ban": "Luat"}\''
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of records to export (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rag_config.yaml"),
        help="Path to the RAG configuration YAML file.",
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
        collection_name=args.collection,
        output_base=args.output.resolve(),
        fmt=args.format,
        where_filter=args.filter,
        limit=args.limit,
        config_path=args.config.resolve(),
    )


if __name__ == "__main__":
    main()
