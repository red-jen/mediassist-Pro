# app/rag/vectorstore.py  —  chunks  →  ChromaDB  /  query  →  results
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from langchain_core.documents import Document
from dotenv import load_dotenv

from .embeddings import get_embedding_service

load_dotenv()


class VectorStore:
    """Store chunks in ChromaDB and search them by semantic similarity."""

    def __init__(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None):
        self.collection_name   = collection_name   or os.getenv("COLLECTION_NAME",   "biomedical_manuals")
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client     = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedder   = get_embedding_service()

        print(f"VectorStore ready — {self.collection.count()} docs in '{self.collection_name}'")

    def add_documents(self, documents: List[Document]) -> None:
        """Embed chunks and persist them in ChromaDB."""
        if not documents:
            return
        texts     = [doc.page_content for doc in documents]
        metadatas = [doc.metadata     for doc in documents]
        ids       = [f"{doc.metadata.get('filename','doc')}_{doc.metadata.get('chunk_id',i)}"
                     for i, doc in enumerate(documents)]
        self.collection.add(
            embeddings=self.embedder.embed_documents(texts),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"  stored {len(documents)} chunks  (total {self.collection.count()})")

    def similarity_search(self, query: str, top_k: int = None,
                          metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return the top_k chunks most relevant to query."""
        top_k     = top_k or int(os.getenv("TOP_K", 5))
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.0))

        results = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            {
                "content":          results["documents"][0][i],
                "metadata":         results["metadatas"][0][i],
                "similarity_score": round(1 - results["distances"][0][i], 4),
                "rank":             i + 1,
            }
            for i in range(len(results["documents"][0]))
        ]
        return [h for h in hits if h["similarity_score"] >= threshold] if threshold > 0 else hits

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "collection_name":     self.collection_name,
            "total_documents":     self.collection.count(),
            "embedding_model":     self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
        }

    def reset_collection(self) -> None:
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)


def create_vector_store(collection_name: str = None) -> VectorStore:
    return VectorStore(collection_name=collection_name)