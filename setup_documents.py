#!/usr/bin/env python3
"""
Process existing PDF and add to vector store
"""
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def process_pdf_to_vectorstore():
    """Process the existing PDF and add to vector store"""
    from src.preprocessing import PDFProcessor
    from app.rag import create_vector_store
    
    print("🤖 Processing PDF document for RAG system...")
    
    # Initialize components
    vector_store = create_vector_store()
    processor = PDFProcessor()
    
    # Find the PDF file
    pdf_dir = "data/pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in data/pdfs/")
        return
        
    pdf_file = pdf_files[0]  # Use the first PDF found
    pdf_path = os.path.join(pdf_dir, pdf_file)
    
    print(f"📄 Processing: {pdf_file}")
    
    try:
        # Process the PDF into chunks
        chunks = processor.process_pdf_file(pdf_path)
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Add to vector store
        vector_store.add_documents(chunks)
        print(f"✅ Added {len(chunks)} chunks to vector store")
        
        # Get stats
        stats = vector_store.get_collection_stats()
        print(f"\n📊 Vector Store Stats:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Embedding model: {stats['embedding_model']}")
        
        print(f"\n🎉 Ready to test RAG system!")
        print(f"   Try asking: 'How do I calibrate laboratory equipment?'")
        print(f"   Or: 'What are the safety protocols for maintenance?'")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    process_pdf_to_vectorstore()