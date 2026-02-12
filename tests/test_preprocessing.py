import pytest
from pathlib import Path
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import PDFProcessor, process_single_pdf
from utils import inspect_chunks, analyze_chunking_quality

def test_basic_processing():
    """Test if we can process a PDF without errors."""
    # You need to add a sample PDF to data/pdfs/
    pdf_path = "data/pdfs/sample_manual.pdf"  # Replace with your PDF
    
    if not Path(pdf_path).exists():
        pytest.skip("No sample PDF found")
    
    chunks = process_single_pdf(pdf_path)
    
    # Basic assertions
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(len(chunk.page_content) > 0 for chunk in chunks), "All chunks should have content"
    assert all("filename" in chunk.metadata for chunk in chunks), "All chunks should have filename metadata"

def test_chunking_parameters():
    """Test different chunking parameters."""
    processor = PDFProcessor()
    
    # Test that parameters are loaded correctly
    assert processor.chunk_size > 0
    assert processor.chunk_overlap >= 0
    assert processor.chunk_overlap < processor.chunk_size

if __name__ == "__main__":
    # Quick manual test
    print("🚀 Testing PDF preprocessing...")
    
    # Look for any PDF in the pdfs directory
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if pdf_files:
        pdf_path = pdf_files[0]  # Use the first PDF found
        print(f"Found PDF: {pdf_path}")
        
        chunks = process_single_pdf(str(pdf_path))
        
        # Inspect results
        inspect_chunks(chunks)
        analyze_chunking_quality(chunks)
        
        print(f"\n✅ Success! Created {len(chunks)} chunks from PDF")
    else:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("   Add your PDF to data/pdfs/ to test")
        print("   Example: data/pdfs/biomedical_manual.pdf")