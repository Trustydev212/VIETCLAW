"""Tests for vietlaw/config.py."""

from __future__ import annotations

from pathlib import Path

from vietlaw.config import (
    EmbeddingConfig,
    LLMConfig,
    RAGConfig,
    RetrievalConfig,
    VectorStoreConfig,
)


class TestRAGConfig:
    def test_default_values(self):
        cfg = RAGConfig()
        assert cfg.embedding.model == "text-embedding-3-large"
        assert cfg.embedding.dimensions == 3072
        assert cfg.vector_store.backend == "chromadb"
        assert cfg.vector_store.collection_name == "vietlaw_laws"
        assert cfg.retrieval.top_k == 10
        assert cfg.llm.provider == "anthropic"

    def test_from_yaml(self, config_path: Path):
        cfg = RAGConfig.from_yaml(config_path)
        assert cfg.embedding.model == "text-embedding-3-large"
        assert cfg.vector_store.distance_metric == "cosine"
        assert cfg.retrieval.reranking is True
        assert cfg.llm.temperature == 0.1

    def test_from_missing_yaml(self, tmp_path: Path):
        cfg = RAGConfig.from_yaml(tmp_path / "nonexistent.yaml")
        # Should return defaults without error.
        assert cfg.embedding.model == "text-embedding-3-large"

    def test_embedding_config_defaults(self):
        c = EmbeddingConfig()
        assert c.batch_size == 100
        assert c.fallback_model == "text-embedding-3-small"

    def test_retrieval_config_defaults(self):
        c = RetrievalConfig()
        assert c.score_threshold == 0.7
        assert c.hybrid_search_enabled is True
        assert c.keyword_weight == 0.3

    def test_vector_store_config_defaults(self):
        c = VectorStoreConfig()
        assert c.persist_directory == "./vectordb/chroma"

    def test_llm_config_defaults(self):
        c = LLMConfig()
        assert c.max_tokens == 4096
