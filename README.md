# VIETCLAW - Kho Dữ Liệu Pháp Luật Việt Nam cho AI/RAG

> Hệ thống Luật pháp Việt Nam đầy đủ, được cấu trúc hóa cho RAG (Retrieval-Augmented Generation) và các ứng dụng AI.

## Giới thiệu

**VIETCLAW** là kho dữ liệu pháp luật Việt Nam được thiết kế đặc biệt cho các hệ thống AI/RAG. Dữ liệu được tổ chức theo cấu trúc markdown có metadata YAML frontmatter, tối ưu hóa cho việc chunking, embedding và truy xuất thông tin pháp lý.

## Đặc điểm nổi bật

- **Dữ liệu có cấu trúc**: Mỗi văn bản pháp luật được tổ chức với YAML frontmatter chứa metadata đầy đủ
- **Tối ưu cho RAG**: Thiết kế chuyên biệt cho chunking theo điều/khoản/mục
- **Cập nhật mới nhất**: Bao gồm luật pháp đến năm 2026
- **Scripts tự động**: Công cụ xử lý, chunking và indexing tự động
- **Phủ rộng**: Từ Hiến pháp đến Nghị định, Thông tư

## Cấu trúc thư mục

```
VIETCLAW/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   ├── rag_config.yaml          # Cấu hình hệ thống RAG
│   └── metadata_schema.json     # Schema metadata văn bản pháp luật
├── data/
│   ├── hien-phap/               # Hiến pháp nước CHXHCN Việt Nam
│   ├── bo-luat/                 # Các Bộ luật (Hình sự, Dân sự, Lao động...)
│   ├── luat/                    # Các Luật chuyên ngành
│   ├── nghi-dinh/               # Nghị định Chính phủ
│   ├── thong-tu/                # Thông tư hướng dẫn
│   └── van-ban-moi-2024-2026/   # Văn bản pháp luật mới nhất
├── scripts/
│   ├── chunk_documents.py       # Script chunking văn bản
│   ├── build_index.py           # Script tạo index
│   ├── validate_metadata.py     # Kiểm tra metadata
│   └── export_embeddings.py     # Xuất embeddings
└── docs/
    └── USAGE.md                 # Hướng dẫn sử dụng
```

## Metadata Schema

Mỗi file markdown đều có YAML frontmatter với các trường:

```yaml
---
tiêu_đề: "Luật Doanh nghiệp"
số_hiệu: "59/2020/QH14"
loại_văn_bản: "Luật"
cơ_quan_ban_hành: "Quốc hội"
ngày_ban_hành: "2020-06-17"
ngày_hiệu_lực: "2021-01-01"
tình_trạng: "Còn hiệu lực"
lĩnh_vực: "Doanh nghiệp"
tags: ["doanh-nghiệp", "thành-lập", "tổ-chức-lại"]
---
```

## Cách sử dụng với RAG

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chunking văn bản

```bash
python scripts/chunk_documents.py --input data/ --output chunks/
```

### 3. Tạo embeddings

```bash
python scripts/export_embeddings.py --chunks chunks/ --model text-embedding-3-large
```

### 4. Tích hợp với LLM

```python
from vietclaw import VietclawRAG

rag = VietclawRAG(index_path="index/", model="claude-opus-4-6")
response = rag.query("Điều kiện thành lập doanh nghiệp là gì?")
print(response)
```

## Phạm vi bao phủ

| Loại văn bản | Số lượng | Mô tả |
|---|---|---|
| Hiến pháp | 1 | Hiến pháp 2013 (120 điều đầy đủ) |
| Bộ luật | 5 | Hình sự, Dân sự, Lao động, TTHS, TTDS |
| Luật chuyên ngành | 17 | Doanh nghiệp, Đầu tư, Đất đai, Thương mại, Nhà ở, Thuế TNCN/GTGT/TNDN, SHTT, ANM, HNGĐ, BVMT, Giáo dục, Cán bộ CC, PCTN, Khiếu nại, Tố cáo |
| Văn bản mới 2024-2026 | 6 | BHXH, Căn cước, Công đoàn, GDĐT, KD BĐS, ATGT đường bộ |
| Nghị định | 3 | NĐ 145/2020 (Lao động), NĐ 01/2021 (ĐKDN), NĐ 100/2019 (Xử phạt GT) |
| Thông tư | 2 | TT 111/2013 (Thuế TNCN), TT 78/2014 (Thuế TNDN) |

## Đóng góp

Chào mừng mọi đóng góp! Vui lòng đọc [USAGE.md](docs/USAGE.md) trước khi tham gia.

## Giấy phép

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Tác giả

- **Trustydev212** - *Người khởi tạo và phát triển*

---

> *"Pháp luật là nền tảng của xã hội văn minh. AI là công cụ để mọi người tiếp cận pháp luật."*
