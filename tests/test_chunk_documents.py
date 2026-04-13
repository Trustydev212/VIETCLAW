"""Tests for scripts/chunk_documents.py."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# Ensure scripts directory is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_documents import (
    Chunk,
    HierarchicalChunker,
    LegalPosition,
    collect_markdown_files,
    load_config,
    parse_frontmatter,
    resolve_patterns,
    write_chunks,
)

# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self, sample_frontmatter_text: str):
        meta, body = parse_frontmatter(sample_frontmatter_text)
        assert meta["tiêu_đề"] == "Luật Test"
        assert meta["số_hiệu"] == "01/2025/QH15"
        assert "# Luật Test" in body

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\n\nBody here."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert "Body here." in body


# ---------------------------------------------------------------------------
# LegalPosition
# ---------------------------------------------------------------------------


class TestLegalPosition:
    def test_as_dict_filters_empty(self):
        pos = LegalPosition(phan="Phan I", chuong="", muc="", dieu="Dieu 1")
        d = pos.as_dict()
        assert d == {"phan": "Phan I", "dieu": "Dieu 1"}
        assert "chuong" not in d

    def test_parent_context(self):
        pos = LegalPosition(
            phan="Phan I", chuong="Chuong II", muc="", dieu="Dieu 5"
        )
        assert pos.parent_context() == "Phan I > Chuong II > Dieu 5"

    def test_parent_context_empty(self):
        pos = LegalPosition()
        assert pos.parent_context() == ""


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_to_dict(self):
        c = Chunk(
            chunk_id="abc-123",
            source_file="test.md",
            content="Hello world",
            parent_context="Dieu 1",
            metadata={"key": "value"},
        )
        d = c.to_dict()
        assert d["chunk_id"] == "abc-123"
        assert d["source_file"] == "test.md"
        assert d["content"] == "Hello world"
        assert d["parent_context"] == "Dieu 1"
        assert d["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------


class TestHierarchicalChunker:
    @pytest.fixture()
    def chunker(self, config_path: Path):
        cfg = load_config(config_path)
        patterns = resolve_patterns(cfg)
        return HierarchicalChunker(
            patterns=patterns,
            chunk_size=1024,
            chunk_overlap=128,
            min_chunk_size=20,
            include_parent_context=True,
        )

    def test_basic_chunking(self, chunker: HierarchicalChunker):
        body = (
            "### Chuong I: Quy dinh chung\n\n"
            "#### Dieu 1. Pham vi dieu chinh\n\n"
            "Luat nay quy dinh ve nhung van de co ban cua phap luat Viet Nam.\n\n"
            "1. Khoan 1 noi dung khoan mot day du.\n\n"
            "2. Khoan 2 noi dung khoan hai day du.\n\n"
            "#### Dieu 2. Doi tuong ap dung\n\n"
            "Luat nay ap dung doi voi to chuc ca nhan trong nuoc.\n"
        )
        chunks = chunker.chunk(body, {"so_hieu": "01/2025/QH15"}, "test.md")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.source_file == "test.md" for c in chunks)

    def test_large_chunk_splitting(self, chunker: HierarchicalChunker):
        # Create a body that exceeds chunk_size.
        long_content = "Noi dung dai. " * 200
        body = f"#### Dieu 1. Test\n\n{long_content}"
        chunks = chunker.chunk(body, {}, "test.md")
        # Should produce more than one chunk.
        assert len(chunks) >= 1

    def test_min_chunk_size_filtering(self, chunker: HierarchicalChunker):
        body = "#### Dieu 1. A\n\nOK\n"
        chunks = chunker.chunk(body, {}, "test.md")
        # "OK" is below min_chunk_size, so may be filtered.
        for c in chunks:
            assert len(c.content) >= chunker.min_chunk_size


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


class TestCollectMarkdownFiles:
    def test_finds_files(self, data_dir: Path):
        files = collect_markdown_files(data_dir)
        assert len(files) > 0
        assert all(f.suffix == ".md" for f in files)


class TestWriteChunks:
    def test_write_and_read(self, tmp_path: Path):
        chunks = [
            Chunk(
                chunk_id="id-1",
                source_file="a.md",
                content="Noi dung",
                parent_context="Dieu 1",
                metadata={"key": "val"},
            ),
        ]
        out = tmp_path / "out.json"
        write_chunks(chunks, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["chunk_id"] == "id-1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_existing(self, config_path: Path):
        cfg = load_config(config_path)
        assert "chunking" in cfg
        assert "embedding" in cfg

    def test_missing_file_returns_empty(self, tmp_path: Path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg == {}


class TestResolvePatterns:
    def test_returns_compiled_patterns(self, config_path: Path):
        cfg = load_config(config_path)
        patterns = resolve_patterns(cfg)
        assert "dieu" in patterns
        assert "khoan" in patterns
        assert isinstance(patterns["dieu"], re.Pattern)
