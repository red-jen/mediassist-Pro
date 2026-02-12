"""
VECTOR STORE MODULE - Persistent Embedding Storage
================================================

LEARNING OBJECTIVES:
1. Understand why we need vector databases for RAG
2. Learn how ChromaDB organizes and searches embeddings
3. See how metadata filtering enhances retrieval
4. Understand persistence vs. in-memory storage

KEY CONCEPTS:
- Vector DB = specialized database for similarity search
- Collections = groups of related documents
- Metadata = structured info attached to each document
- Similarity search = find vectors closest to query vector
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from dotenv import load_dotenv

from .embeddings import get_embedding_service

load_dotenv()

class VectorStore:
    """
    ChromaDB-based vector storage for document embeddings.
    
    WHY CHROMADB?
    - Lightweight and easy to set up
    - Excellent for learning and prototyping
    - Built-in persistence and metadata filtering
    - No complex server setup required
    """
    
    def __init__(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None):
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "biomedical_manuals")
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Initializing ChromaDB...")
        print(f"   Collection: {self.collection_name}")
        print(f"   Persist to: {self.persist_directory}")
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Biomedical equipment manuals and documentation"}
        )
        
        # Initialize embedding service
        self.embedding_service = get_embedding_service()
        
        print(f"✅ ChromaDB initialized with {self.collection.count()} existing documents")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        PROCESS EXPLANATION:
        1. Extract text content from documents
        2. Generate embeddings for all texts
        3. Prepare metadata for each document
        4. Store in ChromaDB with automatic IDs
        """
        if not documents:
            print("⚠️  No documents to add")
            return
        
        print(f"📥 Adding {len(documents)} documents to vector store...")
        
        # Extract texts and prepare data
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate unique IDs for each document
        ids = [f"doc_{i}_{doc.metadata.get('filename', 'unknown')}_{doc.metadata.get('chunk_id', i)}" 
               for i, doc in enumerate(documents)]
        
        # Generate embeddings (this is the expensive part!)
        embeddings = self.embedding_service.embed_documents(texts)
        
        # Add to ChromaDB
        # NOTE: ChromaDB handles the vector indexing automatically
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(documents)} documents to collection '{self.collection_name}'")
        print(f"   Total documents in collection: {self.collection.count()}")
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        HOW IT WORKS:
        1. Convert query to embedding vector
        2. Find top_k most similar document vectors
        3. Return documents with similarity scores
        4. Optionally filter by metadata
        """
        top_k = top_k or int(os.getenv("TOP_K", 5))
        
        print(f"🔍 Searching for: '{query}'")
        print(f"   Retrieving top {top_k} results")
        
        # Convert query to embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,  # Optional metadata filtering
            include=["documents", "metadatas", "distances"]  # What to return
        )
        
        # Process results into a readable format
        search_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'rank': i + 1
            }
            search_results.append(result)
        
        print(f"✅ Found {len(search_results)} relevant documents")
        
        # Filter by similarity threshold if configured
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.0))
        if threshold > 0:
            filtered_results = [r for r in search_results if r['similarity_score'] >= threshold]
            print(f"   After similarity filtering (≥{threshold}): {len(filtered_results)} results")
            return filtered_results
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        count = self.collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension
        }
        
        if count > 0:
            # Sample some documents to show metadata structure
            sample = self.collection.get(limit=3, include=["metadatas"])
            stats["sample_metadata"] = sample["metadatas"]
        
        return stats
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata only (no similarity matching).
        
        USEFUL FOR:
        - Finding all chunks from a specific document
        - Filtering by document type, date, etc.
        - Administrative queries
        """
        results = self.collection.get(
            where=metadata_filter,
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    
    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution!)"""
        print(f"🗑️  Deleting collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        
        # Also remove persisted data
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        print("✅ Collection deleted")
    
    def reset_collection(self) -> None:
        """Clear all documents from collection but keep the collection itself."""
        print(f"🔄 Resetting collection '{self.collection_name}'...")
        
        # Get all document IDs and delete them
        all_docs = self.collection.get(include=[])
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
        
        print("✅ Collection reset (all documents removed)")


# Convenience functions for common operations
def create_vector_store(collection_name: str = None) -> VectorStore:
    """Create a new vector store instance."""
    return VectorStore(collection_name=collection_name)

def test_vector_store():
    """Test the vector store with sample data."""
    print("🧪 TESTING VECTOR STORE:")
    print("=" * 50)
    
    # Create test store
    store = VectorStore(collection_name="test_collection")
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="The centrifuge should be calibrated monthly to ensure accurate results.",
            metadata={"filename": "centrifuge_manual.pdf", "page": 15, "section": "maintenance"}
        ),
        Document(
            page_content="Spectrophotometer wavelength accuracy must be verified daily before use.",
            metadata={"filename": "spectro_manual.pdf", "page": 8, "section": "calibration"}
        ),
        Document(
            page_content="Emergency shutdown procedures for all laboratory equipment.",
            metadata={"filename": "safety_manual.pdf", "page": 3, "section": "emergency"}
        )
    ]
    
    # Add documents
    store.add_documents(sample_docs)
    
    # Test search
    query = "How do I calibrate laboratory equipment?"
    results = store.similarity_search(query, top_k=2)
    
    print(f"\nQuery: '{query}'")
    print("Results:")
    for result in results:
        print(f"  Score: {result['similarity_score']:.3f}")
        print(f"  Content: {result['content'][:100]}...")
        print(f"  Source: {result['metadata']['filename']}")
        print()
    
    # Show collection stats
    stats = store.get_collection_stats()
    print(f"Collection Stats: {stats}")
    
    # Clean up test collection
    store.delete_collection()
    print("🧹 Test collection cleaned up")

if __name__ == "__main__":
    test_vector_store()