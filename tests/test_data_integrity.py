"""
Data integrity tests.

Verifies that all markdown legal documents in the data/ directory have valid
YAML frontmatter and follow the expected structure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


def _all_md_files() -> list[Path]:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    return sorted(data_dir.rglob("*.md"))


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

REQUIRED_FIELDS = [
    "tiêu_đề",
    "số_hiệu",
    "loại_văn_bản",
    "cơ_quan_ban_hành",
    "ngày_ban_hành",
    "ngày_hiệu_lực",
    "tình_trạng",
    "lĩnh_vực",
]


class TestDataIntegrity:
    """Ensure every legal document in data/ has valid structure."""

    @pytest.mark.parametrize("md_path", _all_md_files(), ids=lambda p: p.name)
    def test_has_frontmatter(self, md_path: Path):
        text = md_path.read_text(encoding="utf-8")
        assert text.startswith("---"), f"{md_path.name}: missing YAML frontmatter"
        match = _FRONTMATTER_RE.match(text)
        assert match, f"{md_path.name}: malformed frontmatter block"

    @pytest.mark.parametrize("md_path", _all_md_files(), ids=lambda p: p.name)
    def test_frontmatter_parses(self, md_path: Path):
        text = md_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        if not match:
            pytest.skip("no frontmatter")
        meta = yaml.safe_load(match.group(1))
        assert isinstance(meta, dict), f"{md_path.name}: frontmatter is not a dict"

    @pytest.mark.parametrize("md_path", _all_md_files(), ids=lambda p: p.name)
    def test_required_fields_present(self, md_path: Path):
        text = md_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        if not match:
            pytest.skip("no frontmatter")
        meta = yaml.safe_load(match.group(1))
        missing = [f for f in REQUIRED_FIELDS if f not in meta]
        assert not missing, f"{md_path.name}: missing fields: {missing}"

    @pytest.mark.parametrize("md_path", _all_md_files(), ids=lambda p: p.name)
    def test_has_body_content(self, md_path: Path):
        text = md_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        body = text[match.end():] if match else text
        assert len(body.strip()) > 50, f"{md_path.name}: body content too short"

    @pytest.mark.parametrize("md_path", _all_md_files(), ids=lambda p: p.name)
    def test_date_format(self, md_path: Path):
        text = md_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        if not match:
            pytest.skip("no frontmatter")
        meta = yaml.safe_load(match.group(1))
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for field in ["ngày_ban_hành", "ngày_hiệu_lực"]:
            val = meta.get(field)
            if val is not None:
                assert date_re.match(str(val)), (
                    f"{md_path.name}: {field}={val!r} is not YYYY-MM-DD"
                )
