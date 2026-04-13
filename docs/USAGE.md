# Huong dan su dung VIETLAW

## Tong quan

VIETLAW la kho du lieu phap luat Viet Nam duoc cau truc hoa cho he thong RAG (Retrieval-Augmented Generation). Tai lieu nay huong dan cach su dung du lieu va cac cong cu di kem.

## Yeu cau he thong

- Python 3.9+
- pip (Python package manager)
- OpenAI API key (cho embeddings)
- Toi thieu 4GB RAM
- Toi thieu 2GB dung luong dia

## Cai dat

```bash
# Clone repository
git clone https://github.com/trustydev212/vietlaw.git
cd vietlaw

# Tao virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoac: venv\Scripts\activate  # Windows

# Cai dat dependencies
pip install -r requirements.txt
```

## Cau truc du lieu

### Dinh dang Markdown voi YAML Frontmatter

Moi van ban phap luat duoc luu duoi dang file Markdown (.md) voi metadata YAML frontmatter o dau file:

```markdown
---
ten_van_ban: "Luat Doanh nghiep"
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

# Luat Doanh nghiep

### Chuong I: Nhung quy dinh chung

#### Dieu 1. Pham vi dieu chinh
...
```

### Cau truc phan cap van ban phap luat

```
Phan (Part)           -> ## Phan I, II...
  Chuong (Chapter)    -> ### Chuong I, II...
    Muc (Section)     -> #### Muc 1, 2...
      Dieu (Article)  -> #### Dieu 1, 2...
        Khoan (Clause)  -> 1. 2. 3...
          Diem (Point)  -> a) b) c)...
```

## Quy trinh xu ly RAG

### Buoc 1: Kiem tra du lieu

```bash
python scripts/validate_metadata.py --data-dir data/ --schema config/metadata_schema.json
```

### Buoc 2: Chunking van ban

```bash
python scripts/chunk_documents.py \
  --input data/ \
  --output chunks/ \
  --config config/rag_config.yaml
```

Tham so tuy chinh:
- `--chunk-size`: Kich thuoc chunk (mac dinh: 1024 ky tu)
- `--overlap`: Do overlap giua cac chunks (mac dinh: 128)
- `--strategy`: Chien luoc chunking (hierarchical/fixed_size/semantic)

### Buoc 3: Tao index va embeddings

```bash
# Cai dat bien moi truong
export OPENAI_API_KEY="your-api-key-here"

# Tao index
python scripts/build_index.py \
  --chunks-dir chunks/ \
  --config config/rag_config.yaml
```

### Buoc 4: Xuat embeddings (tuy chon)

```bash
python scripts/export_embeddings.py \
  --collection vietlaw_laws \
  --format parquet \
  --output exports/embeddings.parquet
```

## Tich hop voi ung dung

### Python SDK co ban

```python
import chromadb
import yaml

# Ket noi ChromaDB
client = chromadb.PersistentClient(path="./vectordb/chroma")
collection = client.get_collection("vietlaw_laws")

# Truy van
results = collection.query(
    query_texts=["dieu kien thanh lap doanh nghiep"],
    n_results=5,
    where={"loai_van_ban": "Luat"}
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['so_hieu']}] {meta['ten_van_ban']}")
    print(f"  {doc[:200]}...")
    print()
```

### Tich hop voi LangChain

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

# Khoi tao
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma(
    persist_directory="./vectordb/chroma",
    embedding_function=embeddings,
    collection_name="vietlaw_laws"
)

# Tao retriever voi filter
retriever = vectordb.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"tinh_trang": "Con hieu luc"}
    }
)

# Tao QA chain
llm = ChatAnthropic(model="claude-opus-4-6", temperature=0.1)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Hoi dap
response = qa_chain.invoke(
    "Theo luat Dat dai 2024, dieu kien cap giay chung nhan quyen su dung dat la gi?"
)
print(response["result"])
```

### Tich hop voi Claude API truc tiep

```python
import anthropic
import chromadb

client = anthropic.Anthropic()
chroma = chromadb.PersistentClient(path="./vectordb/chroma")
collection = chroma.get_collection("vietlaw_laws")

def query_vietlaw(question: str) -> str:
    # Buoc 1: Truy xuat van ban lien quan
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    # Buoc 2: Tao context tu ket qua
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\n---\nVan ban: {meta.get('ten_van_ban', 'N/A')}\n"
        context += f"So hieu: {meta.get('so_hieu', 'N/A')}\n"
        context += f"Noi dung: {doc}\n"
    
    # Buoc 3: Goi Claude
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        temperature=0.1,
        system="Ban la tro ly phap ly AI chuyen ve luat phap Viet Nam. "
               "Hay tra loi dua tren cac dieu luat duoc cung cap. "
               "Luon trich dan so hieu van ban va dieu khoan cu the.",
        messages=[{
            "role": "user",
            "content": f"Cau hoi: {question}\n\nTai lieu phap ly lien quan:\n{context}"
        }]
    )
    
    return message.content[0].text

# Su dung
answer = query_vietlaw("Quyen loi cua nguoi lao dong khi bi sa thai trai phap luat?")
print(answer)
```

## Dong gop du lieu

### Them van ban moi

1. Tao file .md trong thu muc phu hop (data/luat/, data/nghi-dinh/...)
2. Them YAML frontmatter day du theo schema
3. Cau truc noi dung theo heading hierarchy
4. Chay validate: `python scripts/validate_metadata.py --data-dir data/`
5. Tao pull request

### Quy uoc dat ten file

```
{loai-van-ban}-{ten-rut-gon}-{nam}.md
```

Vi du:
- `luat-doanh-nghiep-2020.md`
- `nghi-dinh-01-2021-nd-cp.md`
- `thong-tu-111-2013-tt-btc.md`

## FAQ

**Q: Du lieu co duoc cap nhat khong?**
A: Du lieu duoc cap nhat dinh ky theo cac van ban moi ban hanh. Phien ban hien tai bao gom luat phap den 2026.

**Q: Co the su dung cho muc dich thuong mai khong?**
A: Du lieu phap luat la thong tin cong khai. Tuy nhien, hay kiem tra giay phep cu the truoc khi su dung thuong mai.

**Q: Lam sao de bao cao loi trong noi dung?**
A: Mo issue tren GitHub hoac gui pull request voi noi dung chinh xac.

**Q: Co ho tro ngon ngu khac ngoai tieng Viet khong?**
A: Hien tai chi ho tro tieng Viet. Phien ban tieng Anh co the duoc them trong tuong lai.
