"""
Configuration loader for the VIETLAW RAG pipeline.

Reads ``config/rag_config.yaml`` and exposes typed dataclass helpers so the
rest of the package does not need to know about raw YAML keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100
    fallback_model: str = "text-embedding-3-small"
    fallback_dimensions: int = 1536


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"
    collection_name: str = "vietlaw_laws"
    distance_metric: str = "cosine"
    persist_directory: str = "./vectordb/chroma"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    score_threshold: float = 0.7
    reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    hybrid_search_enabled: bool = True
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1
    system_prompt: str = (
        "Ban la tro ly phap ly AI chuyen ve luat phap Viet Nam. "
        "Hay tra loi cau hoi dua tren cac dieu luat duoc cung cap. "
        "Luon trich dan so hieu van ban, dieu khoan cu the. "
        "Neu khong chac chan, hay noi ro va khuyen nguoi dung tham khao luat su."
    )


@dataclass
class RAGConfig:
    """Top-level configuration container."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open(encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        emb_raw = raw.get("embedding", {})
        vs_raw = raw.get("vector_store", {})
        chroma_raw = vs_raw.get("chroma", {})
        ret_raw = raw.get("retrieval", {})
        hybrid_raw = ret_raw.get("hybrid_search", {})
        llm_raw = raw.get("llm", {})

        return cls(
            embedding=EmbeddingConfig(
                model=emb_raw.get("model", EmbeddingConfig.model),
                dimensions=int(emb_raw.get("dimensions", EmbeddingConfig.dimensions)),
                batch_size=int(emb_raw.get("batch_size", EmbeddingConfig.batch_size)),
                fallback_model=emb_raw.get("fallback_model", EmbeddingConfig.fallback_model),
                fallback_dimensions=int(
                    emb_raw.get("fallback_dimensions", EmbeddingConfig.fallback_dimensions)
                ),
            ),
            vector_store=VectorStoreConfig(
                backend=vs_raw.get("backend", VectorStoreConfig.backend),
                collection_name=vs_raw.get(
                    "collection_name", VectorStoreConfig.collection_name
                ),
                distance_metric=vs_raw.get(
                    "distance_metric", VectorStoreConfig.distance_metric
                ),
                persist_directory=chroma_raw.get(
                    "persist_directory", VectorStoreConfig.persist_directory
                ),
            ),
            retrieval=RetrievalConfig(
                top_k=int(ret_raw.get("top_k", RetrievalConfig.top_k)),
                score_threshold=float(
                    ret_raw.get("score_threshold", RetrievalConfig.score_threshold)
                ),
                reranking=bool(ret_raw.get("reranking", RetrievalConfig.reranking)),
                reranker_model=ret_raw.get(
                    "reranker_model", RetrievalConfig.reranker_model
                ),
                hybrid_search_enabled=bool(
                    hybrid_raw.get("enabled", RetrievalConfig.hybrid_search_enabled)
                ),
                keyword_weight=float(
                    hybrid_raw.get("keyword_weight", RetrievalConfig.keyword_weight)
                ),
                semantic_weight=float(
                    hybrid_raw.get("semantic_weight", RetrievalConfig.semantic_weight)
                ),
            ),
            llm=LLMConfig(
                provider=llm_raw.get("provider", LLMConfig.provider),
                model=llm_raw.get("model", LLMConfig.model),
                max_tokens=int(llm_raw.get("max_tokens", LLMConfig.max_tokens)),
                temperature=float(
                    llm_raw.get("temperature", LLMConfig.temperature)
                ),
                system_prompt=llm_raw.get(
                    "system_prompt", LLMConfig.system_prompt
                ),
            ),
        )
