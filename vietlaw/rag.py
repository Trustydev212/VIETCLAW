"""
High-level RAG interface for Vietnamese legal question answering.

This is the main entry point referenced in the README::

    from vietlaw import VietlawRAG

    rag = VietlawRAG(index_path="index/")
    answer = rag.query("Dieu kien thanh lap doanh nghiep la gi?")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vietlaw.config import RAGConfig
from vietlaw.retriever import RetrievalResult, VietlawRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def _call_anthropic(
    prompt: str,
    context: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Cau hoi: {prompt}\n\n"
                    f"Tai lieu phap ly lien quan:\n{context}"
                ),
            }
        ],
    )
    return message.content[0].text


def _call_openai(
    prompt: str,
    context: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the OpenAI Chat Completions API."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Cau hoi: {prompt}\n\n"
                    f"Tai lieu phap ly lien quan:\n{context}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


_LLM_BACKENDS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


# ---------------------------------------------------------------------------
# VietlawRAG
# ---------------------------------------------------------------------------


class VietlawRAG:
    """
    End-to-end Retrieval-Augmented Generation for Vietnamese law.

    Parameters
    ----------
    index_path : str | Path | None
        Path to the ChromaDB persist directory.  Overrides config value.
    config_path : str | Path | None
        Path to ``rag_config.yaml``.
    model : str | None
        LLM model override (e.g. ``"claude-sonnet-4-20250514"``).
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        config_path: str | Path | None = None,
        model: str | None = None,
    ) -> None:
        cfg_path = config_path or "config/rag_config.yaml"
        self._config = RAGConfig.from_yaml(cfg_path)

        if index_path:
            self._config.vector_store.persist_directory = str(index_path)
        if model:
            self._config.llm.model = model

        self._retriever = VietlawRetriever(config=self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> RAGConfig:
        return self._config

    @property
    def retriever(self) -> VietlawRetriever:
        return self._retriever

    def query(
        self,
        question: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> str:
        """
        Answer a legal question using RAG.

        1. Retrieve relevant chunks from the vector store.
        2. Build a context string with citations.
        3. Call the configured LLM to generate an answer.

        Returns the LLM's answer as a plain string.
        """
        results = self._retriever.search(
            query=question, top_k=top_k, where=where
        )

        if not results:
            return (
                "Khong tim thay tai lieu phap ly lien quan den cau hoi cua ban. "
                "Vui long thu lai voi cau hoi cu the hon hoac kiem tra lai du lieu."
            )

        context = self._build_context(results)
        llm_cfg = self._config.llm

        backend = _LLM_BACKENDS.get(llm_cfg.provider)
        if backend is None:
            raise ValueError(
                f"Unsupported LLM provider: {llm_cfg.provider!r}. "
                f"Supported: {list(_LLM_BACKENDS)}"
            )

        logger.info(
            "Calling %s/%s with %d context chunks.",
            llm_cfg.provider,
            llm_cfg.model,
            len(results),
        )

        return backend(
            prompt=question,
            context=context,
            system_prompt=llm_cfg.system_prompt,
            model=llm_cfg.model,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
        )

    def search(
        self,
        question: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks without calling the LLM.

        Useful for inspection, debugging, or building custom prompts.
        """
        return self._retriever.search(
            query=question, top_k=top_k, where=where
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM."""
        sections: list[str] = []
        for i, r in enumerate(results, 1):
            citation = r.format_citation()
            section = (
                f"--- Tai lieu {i} ---\n"
                f"Nguon: {citation}\n"
            )
            if r.parent_context:
                section += f"Vi tri: {r.parent_context}\n"
            section += f"Noi dung:\n{r.content}\n"
            sections.append(section)
        return "\n".join(sections)
