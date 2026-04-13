"""Tests for scripts/validate_metadata.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_metadata import (
    ValidationReport,
    build_so_hieu_index,
    check_cross_references,
    load_schema,
    parse_frontmatter,
    validate_document,
    validate_field,
)

# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid(self, sample_frontmatter_text: str):
        meta = parse_frontmatter(sample_frontmatter_text)
        assert meta is not None
        assert meta["tiêu_đề"] == "Luật Test"

    def test_no_frontmatter(self):
        assert parse_frontmatter("# Just a heading") is None

    def test_invalid_yaml(self):
        text = "---\n[invalid yaml\n---\n\nBody"
        with pytest.raises(ValueError, match="YAML parse error"):
            parse_frontmatter(text)


# ---------------------------------------------------------------------------
# validate_field
# ---------------------------------------------------------------------------


class TestValidateField:
    def test_valid_string(self):
        report = ValidationReport()
        validate_field("test", "hello", {"type": "string"}, "f.md", report)
        assert not report.has_errors

    def test_invalid_type(self):
        report = ValidationReport()
        validate_field("test", 123, {"type": "string"}, "f.md", report)
        assert report.has_errors

    def test_valid_enum(self):
        report = ValidationReport()
        validate_field(
            "status",
            "Còn hiệu lực",
            {"type": "string", "enum": ["Còn hiệu lực", "Hết hiệu lực"]},
            "f.md",
            report,
        )
        assert not report.has_errors

    def test_invalid_enum(self):
        report = ValidationReport()
        validate_field(
            "status",
            "Invalid",
            {"type": "string", "enum": ["Còn hiệu lực", "Hết hiệu lực"]},
            "f.md",
            report,
        )
        assert report.has_errors

    def test_date_format_valid(self):
        report = ValidationReport()
        validate_field(
            "date", "2025-01-15", {"type": "string", "format": "date"}, "f.md", report
        )
        assert not report.has_errors

    def test_date_format_invalid(self):
        report = ValidationReport()
        validate_field(
            "date", "15/01/2025", {"type": "string", "format": "date"}, "f.md", report
        )
        assert report.has_errors

    def test_null_allowed(self):
        report = ValidationReport()
        validate_field(
            "field", None, {"type": ["string", "null"]}, "f.md", report
        )
        assert not report.has_errors

    def test_null_not_allowed(self):
        report = ValidationReport()
        validate_field("field", None, {"type": "string"}, "f.md", report)
        assert report.has_errors


# ---------------------------------------------------------------------------
# validate_document
# ---------------------------------------------------------------------------


class TestValidateDocument:
    def test_valid_document(self, sample_metadata: dict, schema_path: Path):
        schema = load_schema(schema_path)
        report = ValidationReport()
        validate_document(sample_metadata, schema, "test.md", report)
        assert not report.has_errors

    def test_missing_required_field(self, sample_metadata: dict, schema_path: Path):
        schema = load_schema(schema_path)
        del sample_metadata["số_hiệu"]
        report = ValidationReport()
        validate_document(sample_metadata, schema, "test.md", report)
        assert report.has_errors

    def test_extra_field_warning(self, sample_metadata: dict, schema_path: Path):
        schema = load_schema(schema_path)
        sample_metadata["unknown_field"] = "value"
        report = ValidationReport()
        validate_document(sample_metadata, schema, "test.md", report)
        assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------


class TestCrossReferences:
    def test_valid_references(self):
        all_meta = {
            "a.md": {"so_hieu": "01/2025/QH15", "van_ban_lien_quan": ["02/2025/QH15"]},
            "b.md": {"so_hieu": "02/2025/QH15"},
        }
        known = build_so_hieu_index(all_meta)
        report = ValidationReport()
        check_cross_references(all_meta, known, report)
        assert len(report.warnings) == 0

    def test_broken_reference(self):
        all_meta = {
            "a.md": {"so_hieu": "01/2025/QH15", "van_ban_lien_quan": ["99/9999/XX"]},
        }
        known = build_so_hieu_index(all_meta)
        report = ValidationReport()
        check_cross_references(all_meta, known, report)
        assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


class TestLoadSchema:
    def test_loads_valid_schema(self, schema_path: Path):
        schema = load_schema(schema_path)
        assert "properties" in schema
        assert "required" in schema

    def test_missing_schema_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            load_schema(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_initially_no_errors(self):
        r = ValidationReport()
        assert not r.has_errors

    def test_add_error(self):
        r = ValidationReport()
        r.add_error("f.md", "field", "msg")
        assert r.has_errors
        assert len(r.errors) == 1

    def test_add_warning(self):
        r = ValidationReport()
        r.add_warning("f.md", "field", "msg")
        assert not r.has_errors
        assert len(r.warnings) == 1
