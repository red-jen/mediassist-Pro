"""
RETRIEVAL MODULE - Smart Document Retrieval
==========================================

LEARNING OBJECTIVES:
1. Understand different retrieval strategies (naive, query expansion, reranking)
2. Learn why hybrid approaches work better than simple similarity search
3. See how to filter and rank retrieved documents
4. Understand the trade-off between precision and recall

KEY CONCEPTS:
- Retrieval = finding relevant documents for a query
- Query expansion = generating alternative query phrasings
- Reranking = reordering results using a different scoring method
- Context window = maximum text that can fit in LLM input
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from .vectorstore import VectorStore
from .embeddings import get_embedding_service

load_dotenv()

@dataclass
class RetrievedDocument:
    """
    Structure for retrieved documents with all necessary info.
    """
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int
    source_info: str = ""  # Human-readable source description
    
    def __post_init__(self):
        """Generate source info after initialization."""
        filename = self.metadata.get('filename', 'Unknown')
        pages = self.metadata.get('likely_pages', [])
        page_info = f"pages {'-'.join(map(str, pages))}" if pages else "unknown page"
        self.source_info = f"{filename} ({page_info})"

class SmartRetriever:
    """
    Advanced document retriever with multiple strategies.
    
    WHY MULTIPLE STRATEGIES?
    - Simple similarity search often misses relevant documents
    - Query expansion catches documents with different terminology
    - Reranking improves relevance of final results
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embedding_service = get_embedding_service()
        
        # Configuration from environment
        self.top_k = int(os.getenv("TOP_K", 10))  # Initial retrieval amount
        self.final_k = 5  # Final number to return to LLM
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
        
        print(f"🔍 SmartRetriever initialized:")
        print(f"   Initial retrieval: {self.top_k} documents")
        print(f"   Final selection: {self.final_k} documents")
        print(f"   Similarity threshold: {self.similarity_threshold}")
    
    def retrieve(self, query: str, strategy: str = "hybrid") -> List[RetrievedDocument]:
        """
        Main retrieval method with different strategies.
        
        STRATEGIES:
        - naive: Simple similarity search
        - expanded: Query expansion + similarity search
        - hybrid: Expanded search + reranking (recommended)
        """
        print(f"🔍 Retrieving documents for: '{query}'")
        print(f"   Strategy: {strategy}")
        
        if strategy == "naive":
            return self._naive_retrieval(query)
        elif strategy == "expanded":
            return self._expanded_retrieval(query)
        elif strategy == "hybrid":
            return self._hybrid_retrieval(query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def _naive_retrieval(self, query: str) -> List[RetrievedDocument]:
        """
        Simple similarity search - baseline approach.
        
        PROS: Fast, simple, direct
        CONS: May miss relevant docs with different terminology
        """
        print("   Using naive similarity search...")
        
        results = self.vector_store.similarity_search(
            query=query,
            top_k=self.final_k
        )
        
        return [self._format_result(result) for result in results]
    
    def _expanded_retrieval(self, query: str) -> List[RetrievedDocument]:
        """
        Query expansion + similarity search.
        
        HOW IT WORKS:
        1. Generate alternative query phrasings
        2. Search with multiple queries
        3. Combine and deduplicate results
        4. Rank by best similarity score
        """
        print("   Using query expansion...")
        
        # Generate expanded queries
        expanded_queries = self._expand_query(query)
        print(f"   Generated {len(expanded_queries)} query variations")
        
        # Search with each expanded query
        all_results = []
        for expanded_query in expanded_queries:
            results = self.vector_store.similarity_search(
                query=expanded_query,
                top_k=self.top_k // len(expanded_queries)  # Split budget across queries
            )
            all_results.extend(results)
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(all_results)
        ranked_results = sorted(unique_results, key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top results
        final_results = ranked_results[:self.final_k]
        return [self._format_result(result) for result in final_results]
    
    def _hybrid_retrieval(self, query: str) -> List[RetrievedDocument]:
        """
        Expanded retrieval + reranking - best approach.
        
        PROCESS:
        1. Expand query to catch more relevant documents
        2. Retrieve more candidates than needed (top_k)
        3. Rerank candidates using cross-encoder or other method
        4. Return top-ranked results
        """
        print("   Using hybrid retrieval (expanded + reranked)...")
        
        # Step 1: Expanded retrieval with more candidates
        expanded_queries = self._expand_query(query)
        all_results = []
        
        for expanded_query in expanded_queries:
            results = self.vector_store.similarity_search(
                query=expanded_query,
                top_k=self.top_k
            )
            all_results.extend(results)
        
        # Step 2: Deduplicate
        unique_results = self._deduplicate_results(all_results)
        print(f"   Found {len(unique_results)} unique candidates")
        
        # Step 3: Rerank (simple reranking based on query-document similarity)
        reranked_results = self._rerank_results(query, unique_results)
        
        # Step 4: Apply threshold filtering
        filtered_results = [
            r for r in reranked_results 
            if r['similarity_score'] >= self.similarity_threshold
        ]
        
        if not filtered_results:
            print(f"   ⚠️  No results above threshold {self.similarity_threshold}, returning top results")
            filtered_results = reranked_results
        
        # Return final selection
        final_results = filtered_results[:self.final_k]
        print(f"   Final selection: {len(final_results)} documents")
        
        return [self._format_result(result) for result in final_results]
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Generate expanded query variations.
        
        SIMPLE APPROACH (can be enhanced with LLM later):
        - Original query
        - Technical variations
        - Synonym substitutions
        """
        expanded_queries = [query]  # Always include original
        
        # Technical term expansions for biomedical domain
        technical_expansions = {
            "calibrat": ["adjust", "set up", "configure", "tune"],
            "maintain": ["service", "repair", "upkeep", "care"],
            "troubleshoot": ["diagnose", "fix", "solve", "debug"],
            "equipment": ["device", "instrument", "apparatus", "machine"],
            "procedure": ["method", "process", "steps", "protocol"],
            "error": ["problem", "issue", "fault", "malfunction"]
        }
        
        query_lower = query.lower()
        for term, synonyms in technical_expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:4]  # Limit to prevent too many queries
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents based on content similarity.
        
        APPROACH: If two results have >95% content overlap, keep the higher-scoring one.
        """
        if not results:
            return []
        
        unique_results = []
        seen_contents = []
        
        # Sort by similarity score first
        sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        
        for result in sorted_results:
            content = result['content']
            is_duplicate = False
            
            # Check against existing results
            for seen_content in seen_contents:
                overlap = self._content_overlap(content, seen_content)
                if overlap > 0.95:  # 95% similarity = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(content)
        
        return unique_results
    
    def _content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap between two texts."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using additional scoring methods.
        
        SIMPLE RERANKING (can be enhanced with cross-encoders):
        - Combine similarity score with metadata relevance
        - Boost results with query terms in metadata
        - Prefer recent documents if available
        """
        for result in results:
            # Start with similarity score
            score = result['similarity_score']
            
            # Boost score if query terms appear in metadata
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', '').lower()
            
            query_terms = query.lower().split()
            metadata_boost = 0
            
            for term in query_terms:
                if term in filename:
                    metadata_boost += 0.1  # Small boost for filename matches
            
            # Apply boost (cap at 1.0)
            result['similarity_score'] = min(1.0, score + metadata_boost)
        
        # Sort by updated scores
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def _format_result(self, result: Dict[str, Any]) -> RetrievedDocument:
        """Convert raw search result to structured format."""
        return RetrievedDocument(
            content=result['content'],
            metadata=result['metadata'],
            similarity_score=result['similarity_score'],
            rank=result.get('rank', 0)
        )
    
    def get_context_for_llm(self, retrieved_docs: List[RetrievedDocument]) -> str:
        """
        Format retrieved documents for LLM input.
        
        IMPORTANT: Structure matters for LLM performance!
        - Clear document boundaries
        - Source attribution
        - Numbered for easy reference
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        context_parts.append("RELEVANT DOCUMENTATION:\n")
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}] Source: {doc.source_info}")
            context_parts.append(f"Relevance: {doc.similarity_score:.2f}")
            context_parts.append(f"Content: {doc.content}")
            context_parts.append("-" * 80)
        
        return "\n".join(context_parts)

# Factory function for easy initialization
def create_retriever(vector_store: VectorStore) -> SmartRetriever:
    """Create a smart retriever instance."""
    return SmartRetriever(vector_store)

def test_retriever():
    """Test the retrieval system with sample data."""
    print("🧪 TESTING SMART RETRIEVER:")
    print("=" * 50)
    
    from .vectorstore import VectorStore
    from langchain.schema import Document
    
    # Create test data
    store = VectorStore(collection_name="test_retrieval")
    
    sample_docs = [
        Document(
            page_content="The centrifuge requires daily calibration checks to maintain accuracy. Follow the calibration procedure in section 4.2.",
            metadata={"filename": "centrifuge_manual.pdf", "section": "calibration", "likely_pages": [15]}
        ),
        Document(
            page_content="Spectrophotometer wavelength verification must be performed before each measurement session.",
            metadata={"filename": "spectro_guide.pdf", "section": "setup", "likely_pages": [8]}
        ),
        Document(
            page_content="Emergency shutdown: Press the red button and wait for complete power down before maintenance.",
            metadata={"filename": "safety_manual.pdf", "section": "emergency", "likely_pages": [3]}
        ),
        Document(
            page_content="Regular maintenance schedule: calibrate instruments monthly, clean weekly, inspect quarterly.",
            metadata={"filename": "maintenance_schedule.pdf", "section": "schedule", "likely_pages": [1]}
        )
    ]
    
    store.add_documents(sample_docs)
    
    # Test retrieval
    retriever = SmartRetriever(store)
    query = "How do I calibrate laboratory equipment?"
    
    print(f"\nQuery: '{query}'")
    print("\nTesting different strategies:")
    
    for strategy in ["naive", "expanded", "hybrid"]:
        print(f"\n--- {strategy.upper()} STRATEGY ---")
        results = retriever.retrieve(query, strategy=strategy)
        
        for result in results:
            print(f"Score: {result.similarity_score:.3f} | {result.source_info}")
            print(f"Content: {result.content[:100]}...")
            print()
    
    # Test context formatting
    hybrid_results = retriever.retrieve(query, strategy="hybrid")
    context = retriever.get_context_for_llm(hybrid_results)
    print("\nFORMATTED CONTEXT FOR LLM:")
    print(context[:500] + "...")
    
    # Cleanup
    store.delete_collection()
    print("\n🧹 Test data cleaned up")

if __name__ == "__main__":
    test_retriever()