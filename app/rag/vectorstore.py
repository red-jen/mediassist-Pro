# =============================================================
# app/rag/vectorstore.py
# -------------------------------------------------------------
# JOB: store Document chunks in ChromaDB,
#      and search for the most relevant ones by similarity.
#
# This is the ONLY place in the project that talks to ChromaDB.
# It does two things:
#   - add_documents(chunks)  → embed + persist
#   - similarity_search(query) → find the closest chunks
#
# Nothing else lives here — no PDF reading, no LLM calls.
# =============================================================

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
    """
    Wraps ChromaDB so the rest of the app never imports chromadb directly.

    Two core methods:
        add_documents(chunks)      — called once by setup_documents.py
        similarity_search(query)   — called on every user question
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.collection_name   = collection_name   or os.getenv("COLLECTION_NAME",   "biomedical_manuals")
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # ChromaDB client — data is saved to disk so it survives restarts
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # Embedding model — converts text → numbers (vectors)
        self.embedding_service = get_embedding_service()

        print(f"✅ VectorStore ready — {self.collection.count()} docs in '{self.collection_name}'")

    # ── Write ─────────────────────────────────────────────────
    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed each chunk and save it to ChromaDB.
        Called by setup_documents.py, not by the query path.
        """
        if not documents:
            return

        texts     = [doc.page_content for doc in documents]
        metadatas = [doc.metadata     for doc in documents]
        ids       = [
            f"{doc.metadata.get('filename', 'doc')}_{doc.metadata.get('chunk_id', i)}"
            for i, doc in enumerate(documents)
        ]

        # embed_documents turns a list of strings into a list of float vectors
        embeddings = self.embedding_service.embed_documents(texts)

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"📥 Stored {len(documents)} chunks → total {self.collection.count()}")

    # ── Read ──────────────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the top_k chunks most relevant to `query`.

        Returns a list of dicts:
            { content, metadata, similarity_score, rank }
        """
        top_k     = top_k or int(os.getenv("TOP_K", 5))
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.0))

        # 1. Embed the question the same way we embedded the chunks
        query_embedding = self.embedding_service.embed_query(query)

        # 2. Ask ChromaDB for the nearest vectors
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 3. Convert distance (lower = closer) to similarity (higher = better)
        hits = [
            {
                "content":          results["documents"][0][i],
                "metadata":         results["metadatas"][0][i],
                "similarity_score": round(1 - results["distances"][0][i], 4),
                "rank":             i + 1,
            }
            for i in range(len(results["documents"][0]))
        ]

        # 4. Optional: drop results below the similarity threshold
        if threshold > 0:
            hits = [h for h in hits if h["similarity_score"] >= threshold]

        return hits

    # ── Stats / admin ─────────────────────────────────────────
    def get_collection_stats(self) -> Dict[str, Any]:
        """Quick summary — used by setup_documents.py and the /health endpoint."""
        return {
            "collection_name":    self.collection_name,
            "total_documents":    self.collection.count(),
            "persist_directory":  self.persist_directory,
            "embedding_model":    self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
        }

    def reset_collection(self) -> None:
        """Remove all documents (keeps the collection itself)."""
        all_ids = self.collection.get(include=[])["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        print(f"🔄 Collection '{self.collection_name}' cleared.")

    def delete_collection(self) -> None:
        """Wipe the collection and its on-disk data entirely."""
        self.client.delete_collection(self.collection_name)
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        print(f"🗑️  Collection '{self.collection_name}' deleted.")


# ── Factory ───────────────────────────────────────────────────
def create_vector_store(collection_name: str = None) -> VectorStore:
    """
    Single entry point used everywhere in the project.
    Avoids scattering VectorStore(...) calls across files.
    """
    return VectorStore(collection_name=collection_name)