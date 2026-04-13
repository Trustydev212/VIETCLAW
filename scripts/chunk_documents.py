#!/usr/bin/env python3
"""
chunk_documents.py - Hierarchical chunking of Vietnamese legal documents.

Parses markdown files with YAML frontmatter from the data/ directory and
splits them into structured chunks following Vietnamese legal hierarchy:
    Phan (Part) > Chuong (Chapter) > Muc (Section) >
    Dieu (Article) > Khoan (Clause) > Diem (Point)

Each output chunk carries:
  - The text content of the chunk
  - Metadata from the document frontmatter
  - Positional metadata (current phan/chuong/muc/dieu headings)
  - An optional parent-context prefix for retrieval grounding

Usage:
    python scripts/chunk_documents.py \
        --input  data/ \
        --output chunks/ \
        --config config/rag_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
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
logger = logging.getLogger("chunk_documents")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LegalPosition:
    """Tracks the current position within a legal document's hierarchy."""

    phan: str = ""       # Phan (Part)
    chuong: str = ""     # Chuong (Chapter)
    muc: str = ""        # Muc (Section)
    dieu: str = ""       # Dieu (Article)

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v}

    def parent_context(self) -> str:
        """Return a compact breadcrumb string for the current position."""
        parts: list[str] = []
        if self.phan:
            parts.append(self.phan)
        if self.chuong:
            parts.append(self.chuong)
        if self.muc:
            parts.append(self.muc)
        if self.dieu:
            parts.append(self.dieu)
        return " > ".join(parts)


@dataclass
class Chunk:
    """A single chunk produced from a legal document."""

    chunk_id: str
    source_file: str
    content: str
    parent_context: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "content": self.content,
            "parent_context": self.parent_context,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    if not config_path.exists():
        logger.warning("Config file not found at %s; using defaults.", config_path)
        return {}
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", config_path)
    return cfg


def resolve_patterns(cfg: dict[str, Any]) -> dict[str, re.Pattern[str]]:
    """
    Compile regex patterns for Vietnamese legal structure headings.

    Falls back to sensible defaults when the config does not define them.
    """
    raw: dict[str, str] = (
        cfg.get("processing", {}).get("legal_patterns", {})
    )
    defaults: dict[str, str] = {
        "phan":   r"^## Phan [IVXLC]+",
        "chuong": r"^## Chuong [IVXLC0-9]+",
        "muc":    r"^### Muc [0-9]+",
        "dieu":   r"^#### Dieu [0-9]+",
        "khoan":  r"^[0-9]+\.",
        "diem":   r"^[a-z]\)",
    }
    defaults.update(raw)  # config values win over defaults

    compiled: dict[str, re.Pattern[str]] = {}
    for level, pattern in defaults.items():
        try:
            compiled[level] = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            logger.error("Bad regex for level '%s': %s – using default.", level, exc)
            compiled[level] = re.compile(defaults[level], re.MULTILINE)
    return compiled


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Split YAML frontmatter from the body of a markdown file.

    Returns ``(metadata_dict, body_text)``.  If no frontmatter is found
    the metadata dict is empty and body_text equals the original text.
    """
    if not text.startswith("---"):
        return {}, text

    # Match the opening --- ... closing --- fence.
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = text[match.end():]

    try:
        metadata: dict[str, Any] = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        logger.warning("YAML parse error in frontmatter: %s", exc)
        metadata = {}

    return metadata, body


# ---------------------------------------------------------------------------
# Hierarchical chunker
# ---------------------------------------------------------------------------

# Heading levels that update the LegalPosition tracker (order matters).
_POSITION_LEVELS = ("phan", "chuong", "muc", "dieu")


class HierarchicalChunker:
    """
    Splits a legal document body into chunks that respect its hierarchy.

    Algorithm
    ---------
    1. The document is split on *Dieu* boundaries (the primary chunk unit).
    2. If a Dieu chunk exceeds ``chunk_size``, it is further split at
       Khoan boundaries, then by raw character overlap windows.
    3. Each chunk records the LegalPosition (phan/chuong/muc/dieu) at the
       point it was emitted, plus an optional parent-context prefix.
    """

    def __init__(
        self,
        patterns: dict[str, re.Pattern[str]],
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        include_parent_context: bool = True,
    ) -> None:
        self.patterns = patterns
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.include_parent_context = include_parent_context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        body: str,
        frontmatter: dict[str, Any],
        source_file: str,
    ) -> list[Chunk]:
        """Produce all chunks for one document."""
        lines = body.splitlines(keepends=True)
        segments = self._split_into_segments(lines)
        chunks: list[Chunk] = []

        for segment_text, position in segments:
            segment_text = segment_text.strip()
            if len(segment_text) < self.min_chunk_size:
                continue

            sub_chunks = self._split_segment(segment_text)
            for idx, text in enumerate(sub_chunks):
                text = text.strip()
                if len(text) < self.min_chunk_size:
                    continue

                metadata = {
                    **frontmatter,
                    **position.as_dict(),
                    "chunk_index": idx,
                    "total_sub_chunks": len(sub_chunks),
                }
                ctx = position.parent_context() if self.include_parent_context else ""
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        source_file=source_file,
                        content=text,
                        parent_context=ctx,
                        metadata=metadata,
                    )
                )

        logger.debug(
            "Produced %d chunks from '%s'.", len(chunks), source_file
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_heading(self, line: str, level: str) -> bool:
        """Return True if *line* matches the pattern for *level*."""
        return bool(self.patterns[level].match(line.rstrip()))

    def _update_position(self, line: str, position: LegalPosition) -> bool:
        """
        Update *position* in-place when *line* is a hierarchy heading.

        Returns True if the line was a heading, False otherwise.
        """
        stripped = line.rstrip()
        for level in _POSITION_LEVELS:
            if self.patterns[level].match(stripped):
                # Reset all child levels when a higher level heading is seen.
                idx = _POSITION_LEVELS.index(level)
                setattr(position, level, stripped.lstrip("#").strip())
                for child in _POSITION_LEVELS[idx + 1 :]:
                    setattr(position, child, "")
                return True
        return False

    def _split_into_segments(
        self, lines: list[str]
    ) -> list[tuple[str, LegalPosition]]:
        """
        Walk *lines* and emit ``(segment_text, snapshot_position)`` pairs.

        A new segment starts at every *Dieu* heading (or at the start of
        the document for the preamble).
        """
        position = LegalPosition()
        segments: list[tuple[str, LegalPosition]] = []
        current: list[str] = []

        for line in lines:
            is_dieu = self.patterns["dieu"].match(line.rstrip())
            is_higher = any(
                self.patterns[lvl].match(line.rstrip())
                for lvl in ("phan", "chuong", "muc")
            )

            if is_dieu:
                # Flush current segment before starting new Dieu.
                if current:
                    segments.append(("".join(current), self._snapshot(position)))
                    current = []
                self._update_position(line, position)
                current.append(line)
            elif is_higher:
                # Flush, then update position tracker (no new content segment).
                if current:
                    segments.append(("".join(current), self._snapshot(position)))
                    current = []
                self._update_position(line, position)
                # Include the heading in the next segment as context.
                current.append(line)
            else:
                current.append(line)

        if current:
            segments.append(("".join(current), self._snapshot(position)))

        return segments

    @staticmethod
    def _snapshot(position: LegalPosition) -> LegalPosition:
        """Return a copy of *position* so the list entry is not mutated later."""
        return LegalPosition(
            phan=position.phan,
            chuong=position.chuong,
            muc=position.muc,
            dieu=position.dieu,
        )

    def _split_segment(self, text: str) -> list[str]:
        """
        Break *text* into sized sub-chunks if it exceeds ``chunk_size``.

        Tries to split on Khoan (numbered clause) boundaries first, then
        falls back to a sliding-window character split with overlap.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Try splitting on Khoan boundaries.
        khoan_parts = self._split_on_pattern(text, self.patterns["khoan"])
        if len(khoan_parts) > 1:
            merged = self._merge_parts(khoan_parts)
            if all(len(p) <= self.chunk_size for p in merged):
                return merged

        # Fallback: sliding character window.
        return self._sliding_window(text)

    def _split_on_pattern(
        self, text: str, pattern: re.Pattern[str]
    ) -> list[str]:
        """Split *text* at every occurrence of *pattern* (boundary-preserving)."""
        positions = [m.start() for m in pattern.finditer(text)]
        if not positions:
            return [text]

        parts: list[str] = []
        prev = 0
        for pos in positions:
            if pos > prev:
                parts.append(text[prev:pos])
            prev = pos
        parts.append(text[prev:])
        return [p for p in parts if p.strip()]

    def _merge_parts(self, parts: list[str]) -> list[str]:
        """
        Greedily merge small parts so each merged chunk fits in ``chunk_size``.
        """
        merged: list[str] = []
        current = ""
        for part in parts:
            if current and len(current) + len(part) > self.chunk_size:
                merged.append(current)
                # Carry overlap from the tail of the previous chunk.
                current = current[-self.chunk_overlap :] + part
            else:
                current += part
        if current:
            merged.append(current)
        return merged

    def _sliding_window(self, text: str) -> list[str]:
        """Character-level sliding window split with configurable overlap."""
        chunks: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            if end == length:
                break
            start = end - self.chunk_overlap
        return chunks


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def collect_markdown_files(input_dir: Path) -> list[Path]:
    """Recursively collect all ``.md`` files under *input_dir*."""
    files = sorted(input_dir.rglob("*.md"))
    logger.info("Found %d markdown files under '%s'.", len(files), input_dir)
    return files


def write_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Serialise *chunks* as a JSON array to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in chunks]
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %d chunks → %s", len(chunks), output_path)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def process_file(
    md_path: Path,
    output_dir: Path,
    chunker: HierarchicalChunker,
) -> int:
    """
    Chunk one markdown file and write the result to *output_dir*.

    Returns the number of chunks produced.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read '%s': %s", md_path, exc)
        return 0

    frontmatter, body = parse_frontmatter(text)
    if not body.strip():
        logger.warning("'%s' has no body content; skipping.", md_path)
        return 0

    relative_path = str(md_path)
    chunks = chunker.chunk(body, frontmatter, source_file=relative_path)

    if not chunks:
        logger.warning("No usable chunks from '%s'.", md_path)
        return 0

    # Mirror the input directory tree under output_dir.
    stem = md_path.stem
    output_path = output_dir / md_path.parent.name / f"{stem}.json"
    write_chunks(chunks, output_path)
    return len(chunks)


def run(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
) -> None:
    """Entry-point for the chunking pipeline."""
    cfg = load_config(config_path)
    chunking_cfg: dict[str, Any] = cfg.get("chunking", {})

    chunk_size: int = int(chunking_cfg.get("chunk_size", 1024))
    chunk_overlap: int = int(chunking_cfg.get("chunk_overlap", 128))
    min_chunk_size: int = int(chunking_cfg.get("min_chunk_size", 100))
    include_parent: bool = bool(chunking_cfg.get("include_parent_context", True))

    patterns = resolve_patterns(cfg)
    chunker = HierarchicalChunker(
        patterns=patterns,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        include_parent_context=include_parent,
    )

    md_files = collect_markdown_files(input_dir)
    if not md_files:
        logger.error("No markdown files found in '%s'. Exiting.", input_dir)
        sys.exit(1)

    total_chunks = 0
    total_files = 0
    errors = 0

    for md_path in md_files:
        try:
            n = process_file(md_path, output_dir, chunker)
            if n:
                total_chunks += n
                total_files += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error processing '%s': %s", md_path, exc)
            errors += 1

    logger.info(
        "Done. Files processed: %d  |  Total chunks: %d  |  Errors: %d",
        total_files,
        total_chunks,
        errors,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk Vietnamese legal markdown documents hierarchically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Root directory containing .md legal documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chunks"),
        help="Directory where JSON chunk files will be written.",
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
        input_dir=args.input.resolve(),
        output_dir=args.output.resolve(),
        config_path=args.config.resolve(),
    )


if __name__ == "__main__":
    main()
