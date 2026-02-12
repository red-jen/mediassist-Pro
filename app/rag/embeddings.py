"""
EMBEDDINGS MODULE - Converting Text to Vectors
=============================================

LEARNING OBJECTIVES:
1. Understand what embeddings are and why we need them
2. Learn how to generate consistent embeddings for documents and queries
3. See how embeddings enable semantic search (not just keyword matching)

KEY CONCEPT:
- Embeddings = numerical representations of text meaning
- Similar texts have similar embeddings (close in vector space)
- This enables "semantic similarity" search beyond exact keyword matches
"""

import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    """
    Handles text-to-vector conversion for the RAG system.
    
    WHY SENTENCE-TRANSFORMERS?
    - Runs locally (no API costs, privacy-friendly)
    - Optimized for semantic search tasks
    - Good balance of speed vs. quality
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", 384))
        
        print(f"🔄 Loading embedding model: {self.model_name}")
        # This downloads the model on first use, then caches it locally
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded! Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert document chunks to embeddings.
        
        WHY BATCH PROCESSING?
        - More efficient than one-by-one conversion
        - Better GPU utilization (if available)
        - Consistent normalization across documents
        """
        print(f"🔄 Embedding {len(texts)} documents...")
        
        # Generate embeddings for all documents at once
        embeddings = self.model.encode(
            texts,
            batch_size=32,           # Process in batches for memory efficiency
            show_progress_bar=True,  # Show progress for large document sets
            convert_to_numpy=True    # Return as numpy arrays
        )
        
        # Convert to list of lists (required by most vector DBs)
        embeddings_list = embeddings.tolist()
        
        print(f"✅ Generated {len(embeddings_list)} embeddings")
        return embeddings_list
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert user query to embedding.
        
        IMPORTANT: Must use the same model as documents!
        Different models create incompatible vector spaces.
        """
        print(f"🔍 Embedding query: '{query[:50]}...'")
        
        embedding = self.model.encode([query])[0]  # Single query
        return embedding.tolist()
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": self.model.max_seq_length,
            "model_size_mb": "~80MB"  # Approximate for MiniLM
        }
    
    def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        COSINE SIMILARITY EXPLAINED:
        - Measures angle between vectors (not distance)
        - Range: -1 (opposite) to 1 (identical)
        - Values > 0.7 usually indicate good semantic similarity
        """
        # Convert to numpy for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity formula: A·B / (|A| * |B|)
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        # Prevent division by zero
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        return float(similarity)

# Singleton pattern for model reuse
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    """
    Get shared embedding service instance.
    
    WHY SINGLETON?
    - Model loading is expensive (1-2 seconds)
    - We only need one model instance in memory
    - Ensures consistent embeddings across the application
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

# Quick testing function
def test_embeddings():
    """Test the embedding service with sample biomedical text."""
    service = get_embedding_service()
    
    # Sample biomedical texts
    texts = [
        "The centrifuge requires regular maintenance every 3 months",
        "Calibration of the spectrophotometer should be performed daily",
        "Equipment malfunction troubleshooting procedures"
    ]
    
    query = "How often should I maintain laboratory equipment?"
    
    print("🧪 TESTING EMBEDDINGS:")
    print("-" * 50)
    
    # Generate embeddings
    doc_embeddings = service.embed_documents(texts)
    query_embedding = service.embed_query(query)
    
    # Calculate similarities
    print(f"Query: '{query}'")
    print("\nSimilarity scores:")
    for i, text in enumerate(texts):
        score = service.similarity_score(query_embedding, doc_embeddings[i])
        print(f"  {score:.3f} - {text}")
    
    print(f"\n📊 Model info: {service.get_model_info()}")

if __name__ == "__main__":
    test_embeddings()