# Dong gop cho VIETLAW

Cam on ban da quan tam den du an VIETLAW! Tai lieu nay huong dan cach dong gop.

## Cach dong gop

### 1. Bao cao loi (Bug Report)

- Mo issue tren GitHub voi tieu de mo ta ngan gon
- Mo ta chi tiet loi gap phai va cac buoc tai hien
- Dinh kem log hoac thong bao loi neu co

### 2. De xuat tinh nang (Feature Request)

- Mo issue voi nhan `enhancement`
- Mo ta tinh nang mong muon va ly do can thiet

### 3. Dong gop code (Pull Request)

#### Thiet lap moi truong phat trien

```bash
# Clone repository
git clone https://github.com/trustydev212/vietlaw.git
cd vietlaw

# Tao virtual environment
python -m venv venv
source venv/bin/activate

# Cai dat dependencies (bao gom dev tools)
pip install -e ".[all]"
```

#### Quy trinh dong gop

1. Fork repository
2. Tao branch moi: `git checkout -b feature/ten-tinh-nang`
3. Thuc hien thay doi
4. Chay tests: `make test`
5. Chay lint: `make lint`
6. Commit voi message ro rang
7. Push va tao Pull Request

#### Quy uoc commit message

```
feat: Them tinh nang moi
fix: Sua loi
docs: Cap nhat tai lieu
data: Them/cap nhat van ban phap luat
refactor: Tai cau truc code
test: Them/cap nhat tests
ci: Cap nhat CI/CD
```

### 4. Dong gop du lieu phap luat

Day la cach dong gop quan trong nhat! De them van ban phap luat moi:

#### Buoc 1: Tao file markdown

Tao file `.md` trong thu muc phu hop:
- `data/luat/` - Luat chuyen nganh
- `data/bo-luat/` - Bo luat
- `data/nghi-dinh/` - Nghi dinh
- `data/thong-tu/` - Thong tu
- `data/van-ban-moi-2024-2026/` - Van ban moi

#### Buoc 2: Dat ten file

```
{loai}-{ten-rut-gon}-{nam}.md
```

Vi du: `luat-doanh-nghiep-2020.md`

#### Buoc 3: Them YAML frontmatter

```yaml
---
tieu_de: "Luat Doanh nghiep"
so_hieu: "59/2020/QH14"
loai_van_ban: "Luat"
co_quan_ban_hanh: "Quoc hoi"
ngay_ban_hanh: "2020-06-17"
ngay_hieu_luc: "2021-01-01"
tinh_trang: "Con hieu luc"
linh_vuc: "Doanh nghiep"
tags: ["doanh-nghiep", "thanh-lap", "to-chuc-lai"]
tom_tat: "Quy dinh ve thanh lap, to chuc quan ly..."
---
```

#### Buoc 4: Cau truc noi dung

Su dung heading hierarchy theo quy uoc:
```markdown
# Ten van ban

### Chuong I: Nhung quy dinh chung

#### Dieu 1. Pham vi dieu chinh

1. Khoan 1...
2. Khoan 2...

a) Diem a...
b) Diem b...
```

#### Buoc 5: Kiem tra metadata

```bash
python scripts/validate_metadata.py --data-dir data/
```

#### Buoc 6: Tao Pull Request

Mo PR va mo ta van ban phap luat da them.

## Tieu chuan code

- Su dung type hints (Python 3.9+)
- Viet docstring cho functions/classes
- Dat ten bien/ham bang tieng Anh, comment co the bang tieng Viet
- Dam bao tat ca tests pass truoc khi tao PR

## Lien he

- GitHub Issues: https://github.com/trustydev212/vietlaw/issues
- Email: (lien he qua GitHub)

Cam on ban da dong gop!
