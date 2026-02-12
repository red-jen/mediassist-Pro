"""
RAG __init__.py - RAG Package Initialization
==========================================

This module provides easy access to all RAG components.
"""

from .embeddings import EmbeddingService, get_embedding_service
from .vectorstore import VectorStore, create_vector_store
from .retrieval import SmartRetriever, RetrievedDocument, create_retriever
from .chain import BiomedicRAGChain, RAGResponse, create_rag_chain

__all__ = [
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    
    # Vector Store
    "VectorStore", 
    "create_vector_store",
    
    # Retrieval
    "SmartRetriever",
    "RetrievedDocument", 
    "create_retriever",
    
    # RAG Chain
    "BiomedicRAGChain",
    "RAGResponse",
    "create_rag_chain"
]