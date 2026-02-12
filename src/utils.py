"""Utility functions for inspection and testing."""

from typing import List
from langchain_core.documents import Document
import json

def inspect_chunks(chunks: List[Document], num_samples: int = 3):
    """Print chunk samples for manual inspection."""
    print(f"\n📊 CHUNK INSPECTION (showing {num_samples}/{len(chunks)} chunks)")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\n🔍 CHUNK {i+1}:")
        print(f"   Size: {len(chunk.page_content)} characters")
        print(f"   Pages: {chunk.metadata['likely_pages']}")
        print(f"   Content preview:")
        print(f"   {chunk.page_content[:300]}...")
        print("-" * 50)

def analyze_chunking_quality(chunks: List[Document]):
    """Analyze chunk size distribution and overlaps."""
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    print(f"\n📈 CHUNKING ANALYSIS:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Average size: {sum(sizes)/len(sizes):.0f} characters")
    print(f"   Min size: {min(sizes)}")
    print(f"   Max size: {max(sizes)}")
    
    # Check for potential issues
    small_chunks = [s for s in sizes if s < 200]
    if small_chunks:
        print(f"   ⚠️  {len(small_chunks)} chunks are very small (<200 chars)")
    
    large_chunks = [s for s in sizes if s > 1200]
    if large_chunks:
        print(f"   ⚠️  {len(large_chunks)} chunks are very large (>1200 chars)")