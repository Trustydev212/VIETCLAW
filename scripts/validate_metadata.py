#!/usr/bin/env python3
"""
validate_metadata.py - Validate YAML frontmatter metadata in Vietnamese legal documents.

Checks every ``.md`` file in the data directory against the JSON schema in
``config/metadata_schema.json`` and reports:

  - Files that are missing a frontmatter block entirely
  - Required fields that are absent
  - Fields whose values violate schema constraints (type, enum, pattern, format)
  - Broken cross-references listed in the ``van_ban_lien_quan`` array
    (i.e. ``so_hieu`` values that do not match any known document)

Exit codes:
  0  All files pass validation.
  1  One or more validation errors were found.

Usage:
    python scripts/validate_metadata.py \
        --data-dir data/ \
        --schema   config/metadata_schema.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
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
logger = logging.getLogger("validate_metadata")


# ---------------------------------------------------------------------------
# Validation result data classes
# ---------------------------------------------------------------------------


@dataclass
class FieldError:
    """One validation failure for a single field in a single file."""

    file: str
    field: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated results for a complete validation run."""

    errors: list[FieldError] = field(default_factory=list)
    warnings: list[FieldError] = field(default_factory=list)

    def add_error(self, file: str, field_name: str, message: str) -> None:
        self.errors.append(FieldError(file=file, field=field_name, message=message))

    def add_warning(self, file: str, field_name: str, message: str) -> None:
        self.warnings.append(FieldError(file=file, field=field_name, message=message))

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def print_summary(self) -> None:
        if self.warnings:
            print(f"\n{'─' * 60}")
            print(f"  WARNINGS ({len(self.warnings)})")
            print(f"{'─' * 60}")
            for w in self.warnings:
                print(f"  [WARN]  {w.file}")
                print(f"          field '{w.field}': {w.message}")

        if self.errors:
            print(f"\n{'─' * 60}")
            print(f"  ERRORS ({len(self.errors)})")
            print(f"{'─' * 60}")
            for e in self.errors:
                print(f"  [ERROR] {e.file}")
                print(f"          field '{e.field}': {e.message}")

        print(f"\n{'═' * 60}")
        if self.has_errors:
            print(
                f"  Result: FAILED  "
                f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
            )
        else:
            print(
                f"  Result: PASSED  "
                f"(0 errors, {len(self.warnings)} warning(s))"
            )
        print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load and return the JSON schema; exit with an error if unavailable."""
    if not schema_path.exists():
        logger.error("Schema file not found: %s", schema_path)
        sys.exit(1)
    with schema_path.open(encoding="utf-8") as fh:
        try:
            schema: dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("Cannot parse schema JSON: %s", exc)
            sys.exit(1)
    logger.info("Schema loaded from %s", schema_path)
    return schema


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    """
    Extract YAML frontmatter from *text*.

    Returns the parsed dict, or ``None`` when no frontmatter block is present.
    """
    if not text.startswith("---"):
        return None
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-field schema validators
# ---------------------------------------------------------------------------

# ISO 8601 date pattern (YYYY-MM-DD).
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_type(
    value: Any,
    type_spec: Any,
    field_name: str,
    file_label: str,
    report: ValidationReport,
) -> bool:
    """
    Validate that *value* conforms to the JSON Schema ``type`` specification.

    *type_spec* may be a single type string or a list (as JSON Schema allows).
    Returns False when the type check fails.
    """
    if isinstance(type_spec, list):
        allowed_types = type_spec
    else:
        allowed_types = [type_spec]

    _map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    for t in allowed_types:
        python_type = _map.get(t)
        if python_type is None:
            continue
        if isinstance(value, python_type):
            return True
        # JSON Schema: boolean is a subtype of integer in Python — exclude it.
        if t == "integer" and isinstance(value, bool):
            continue

    report.add_error(
        file=file_label,
        field_name=field_name,
        message=(
            f"Expected type {allowed_types!r}, "
            f"got {type(value).__name__!r} (value={value!r})."
        ),
    )
    return False


def _check_enum(
    value: Any,
    enum_values: list[Any],
    field_name: str,
    file_label: str,
    report: ValidationReport,
) -> None:
    if value not in enum_values:
        report.add_error(
            file=file_label,
            field_name=field_name,
            message=(
                f"Value {value!r} is not one of the allowed values: "
                f"{enum_values!r}."
            ),
        )


def _check_pattern(
    value: str,
    pattern: str,
    field_name: str,
    file_label: str,
    report: ValidationReport,
) -> None:
    if not re.fullmatch(pattern, value):
        report.add_error(
            file=file_label,
            field_name=field_name,
            message=f"Value {value!r} does not match pattern {pattern!r}.",
        )


def _check_format(
    value: str,
    fmt: str,
    field_name: str,
    file_label: str,
    report: ValidationReport,
) -> None:
    if fmt == "date":
        if not _DATE_PATTERN.fullmatch(value):
            report.add_error(
                file=file_label,
                field_name=field_name,
                message=(
                    f"Value {value!r} is not a valid date (expected YYYY-MM-DD)."
                ),
            )


def validate_field(
    field_name: str,
    value: Any,
    field_schema: dict[str, Any],
    file_label: str,
    report: ValidationReport,
) -> None:
    """Run all applicable schema checks for a single field value."""
    type_spec = field_schema.get("type")
    enum_values = field_schema.get("enum")
    pattern = field_schema.get("pattern")
    fmt = field_schema.get("format")

    # Allow null when type includes "null" — skip further checks for None.
    if value is None:
        if type_spec and "null" not in (
            type_spec if isinstance(type_spec, list) else [type_spec]
        ):
            report.add_error(
                file=file_label,
                field_name=field_name,
                message="Value is null but null is not an allowed type.",
            )
        return

    # Type check.
    if type_spec:
        type_ok = _check_type(value, type_spec, field_name, file_label, report)
        if not type_ok:
            return  # Skip further checks if type is wrong.

    # Enum check (applies to the scalar value, not array items).
    if enum_values is not None:
        _check_enum(value, enum_values, field_name, file_label, report)

    # Pattern check (strings only).
    if pattern and isinstance(value, str):
        _check_pattern(value, pattern, field_name, file_label, report)

    # Format check (strings only).
    if fmt and isinstance(value, str):
        _check_format(value, fmt, field_name, file_label, report)


# ---------------------------------------------------------------------------
# Document-level validator
# ---------------------------------------------------------------------------


def validate_document(
    metadata: dict[str, Any],
    schema: dict[str, Any],
    file_label: str,
    report: ValidationReport,
) -> None:
    """Validate *metadata* against *schema* and record results in *report*."""
    required: list[str] = schema.get("required", [])
    properties: dict[str, Any] = schema.get("properties", {})

    # Check required fields.
    for req_field in required:
        if req_field not in metadata or metadata[req_field] is None:
            report.add_error(
                file=file_label,
                field_name=req_field,
                message="Required field is missing or null.",
            )

    # Validate each present field against its property schema.
    for field_name, value in metadata.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            # Field is not defined in the schema — emit a warning.
            report.add_warning(
                file=file_label,
                field_name=field_name,
                message="Field is not defined in the schema (extra field).",
            )
            continue
        validate_field(field_name, value, field_schema, file_label, report)


# ---------------------------------------------------------------------------
# Cross-reference checker
# ---------------------------------------------------------------------------


def build_so_hieu_index(
    all_metadata: dict[str, dict[str, Any]]
) -> set[str]:
    """Collect every ``so_hieu`` value found across all documents."""
    known: set[str] = set()
    for meta in all_metadata.values():
        so_hieu = meta.get("so_hieu")
        if isinstance(so_hieu, str) and so_hieu:
            known.add(so_hieu)
    return known


def check_cross_references(
    all_metadata: dict[str, dict[str, Any]],
    known_so_hieu: set[str],
    report: ValidationReport,
) -> None:
    """
    For each document, verify that every entry in ``van_ban_lien_quan``
    resolves to a known ``so_hieu`` in the corpus.

    Unknown references are reported as warnings (not errors) because the
    referenced document may simply be absent from the current data directory
    rather than being genuinely broken.
    """
    for file_label, meta in all_metadata.items():
        related: Any = meta.get("van_ban_lien_quan")
        if not related:
            continue
        if not isinstance(related, list):
            # Type error already reported by validate_document; skip here.
            continue
        for ref in related:
            if not isinstance(ref, str):
                report.add_warning(
                    file=file_label,
                    field_name="van_ban_lien_quan",
                    message=f"Non-string reference value: {ref!r}.",
                )
                continue
            if ref not in known_so_hieu:
                report.add_warning(
                    file=file_label,
                    field_name="van_ban_lien_quan",
                    message=(
                        f"Cross-reference '{ref}' does not match any known "
                        "so_hieu in the corpus."
                    ),
                )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def collect_markdown_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.md"))
    logger.info("Found %d markdown files under '%s'.", len(files), data_dir)
    return files


def run(data_dir: Path, schema_path: Path) -> bool:
    """
    Validate all markdown files in *data_dir* against *schema_path*.

    Returns True when all files pass (no errors), False otherwise.
    """
    schema = load_schema(schema_path)
    md_files = collect_markdown_files(data_dir)

    if not md_files:
        logger.warning("No markdown files found in '%s'.", data_dir)
        return True

    report = ValidationReport()
    # file_label → parsed metadata (for cross-reference checks)
    all_metadata: dict[str, dict[str, Any]] = {}

    for md_path in md_files:
        file_label = str(md_path)
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            report.add_error(file_label, "<file>", f"Cannot read file: {exc}")
            continue

        try:
            meta = parse_frontmatter(text)
        except ValueError as exc:
            report.add_error(file_label, "<frontmatter>", str(exc))
            continue

        if meta is None:
            report.add_error(
                file_label,
                "<frontmatter>",
                "No YAML frontmatter block found (file must begin with ---).",
            )
            continue

        validate_document(meta, schema, file_label, report)
        all_metadata[file_label] = meta

    # Cross-reference pass (only on files that parsed successfully).
    known_so_hieu = build_so_hieu_index(all_metadata)
    check_cross_references(all_metadata, known_so_hieu, report)

    # Print summary.
    total = len(md_files)
    passed = total - len({e.file for e in report.errors})
    print(f"\nValidated {total} file(s).  {passed} passed, "
          f"{total - passed} had errors.\n")
    report.print_summary()

    return not report.has_errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate YAML frontmatter in Vietnamese legal markdown files "
            "against the project's JSON schema."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing .md legal document files.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("config/metadata_schema.json"),
        help="Path to the JSON schema for document metadata.",
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

    success = run(
        data_dir=args.data_dir.resolve(),
        schema_path=args.schema.resolve(),
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
