"""
Retrieval layer for VIETLAW.

Wraps ChromaDB collection queries and provides optional cross-encoder
reranking via ``sentence-transformers``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vietlaw.config import RAGConfig, RetrievalConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its score and metadata."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "")

    @property
    def parent_context(self) -> str:
        return self.metadata.get("parent_context", "")

    def format_citation(self) -> str:
        """Return a human-readable citation string."""
        title = self.metadata.get("tiêu_đề", self.metadata.get("tieu_de", ""))
        so_hieu = self.metadata.get("số_hiệu", self.metadata.get("so_hieu", ""))
        dieu = self.metadata.get("dieu", "")
        parts = []
        if title:
            parts.append(title)
        if so_hieu:
            parts.append(f"({so_hieu})")
        if dieu:
            parts.append(f"- {dieu}")
        return " ".join(parts) if parts else self.source_file


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class VietlawRetriever:
    """
    Query the ChromaDB vector store and optionally rerank results.

    Parameters
    ----------
    config : RAGConfig | None
        Full RAG configuration.  When *None* sensible defaults are used.
    config_path : str | Path | None
        Path to a ``rag_config.yaml`` file.  Ignored when *config* is given.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if config is None:
            config = RAGConfig.from_yaml(config_path) if config_path else RAGConfig()

        self._cfg = config
        self._vs_cfg: VectorStoreConfig = config.vector_store
        self._ret_cfg: RetrievalConfig = config.retrieval
        self._collection = self._open_collection()
        self._reranker = self._load_reranker()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _open_collection(self) -> Any:
        import chromadb
        from chromadb.config import Settings

        persist_path = Path(self._vs_cfg.persist_directory)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"ChromaDB persist directory not found: {persist_path}. "
                "Run `python scripts/build_index.py` first."
            )

        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=self._vs_cfg.collection_name)
        logger.info(
            "Opened collection '%s' (%d docs).",
            self._vs_cfg.collection_name,
            collection.count(),
        )
        return collection

    def _load_reranker(self) -> Any | None:
        if not self._ret_cfg.reranking:
            return None
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self._ret_cfg.reranker_model)
            logger.info("Loaded reranker: %s", self._ret_cfg.reranker_model)
            return model
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; reranking disabled."
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant legal document chunks.

        Parameters
        ----------
        query : str
            Natural-language query in Vietnamese.
        top_k : int | None
            Override the configured ``top_k``.
        where : dict | None
            ChromaDB metadata filter (``where`` clause).
        score_threshold : float | None
            Override the configured ``score_threshold``.

        Returns
        -------
        list[RetrievalResult]
            Ranked results, highest relevance first.
        """
        k = top_k or self._ret_cfg.top_k
        threshold = score_threshold or self._ret_cfg.score_threshold

        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: list[RetrievalResult] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance is in [0, 2]; convert to similarity.
            score = 1.0 - dist
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=doc or "",
                    score=score,
                    metadata=meta or {},
                )
            )

        # Rerank if available.
        if self._reranker and results:
            results = self._rerank(query, results)

        # Apply score threshold.
        results = [r for r in results if r.score >= threshold]

        return results

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        pairs = [(query, r.content) for r in results]
        scores = self._reranker.predict(pairs)
        for result, score in zip(results, scores):
            result.score = float(score)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
