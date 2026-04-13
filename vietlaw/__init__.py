"""
VIETLAW - Kho Du Lieu Phap Luat Viet Nam cho AI/RAG.

Provides the :class:`VietlawRAG` high-level interface for querying
Vietnamese legal documents backed by a ChromaDB vector store.
"""

from vietlaw.config import RAGConfig
from vietlaw.rag import VietlawRAG
from vietlaw.retriever import VietlawRetriever

__all__ = ["VietlawRAG", "VietlawRetriever", "RAGConfig"]
__version__ = "0.1.0"
