"""Shared pytest fixtures for VIETLAW tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# Root of the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture()
def config_path(project_root: Path) -> Path:
    return project_root / "config" / "rag_config.yaml"


@pytest.fixture()
def schema_path(project_root: Path) -> Path:
    return project_root / "config" / "metadata_schema.json"


@pytest.fixture()
def sample_frontmatter_text() -> str:
    return (
        '---\n'
        'tiêu_đề: "Luật Test"\n'
        'số_hiệu: "01/2025/QH15"\n'
        'loại_văn_bản: "Luật"\n'
        'cơ_quan_ban_hành: "Quốc hội"\n'
        'ngày_ban_hành: "2025-01-01"\n'
        'ngày_hiệu_lực: "2025-07-01"\n'
        'tình_trạng: "Còn hiệu lực"\n'
        'lĩnh_vực: "Test"\n'
        'tags: ["test"]\n'
        '---\n\n'
        '# Luật Test\n\n'
        '### Chương I: Quy định chung\n\n'
        '#### Điều 1. Phạm vi điều chỉnh\n\n'
        'Luật này quy định về test.\n\n'
        '1. Khoản 1 nội dung.\n\n'
        '2. Khoản 2 nội dung.\n\n'
        'a) Điểm a nội dung.\n\n'
        'b) Điểm b nội dung.\n\n'
        '#### Điều 2. Đối tượng áp dụng\n\n'
        'Luật này áp dụng đối với:\n\n'
        '1. Tổ chức, cá nhân trong nước.\n\n'
        '2. Tổ chức, cá nhân nước ngoài hoạt động tại Việt Nam.\n'
    )


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "tiêu_đề": "Luật Test",
        "số_hiệu": "01/2025/QH15",
        "loại_văn_bản": "Luật",
        "cơ_quan_ban_hành": "Quốc hội",
        "ngày_ban_hành": "2025-01-01",
        "ngày_hiệu_lực": "2025-07-01",
        "tình_trạng": "Còn hiệu lực",
        "lĩnh_vực": "Test",
        "tags": ["test"],
    }
