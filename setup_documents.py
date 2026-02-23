
# =============================================================
# setup_documents.py
# -------------------------------------------------------------
# JOB: find every PDF in data/pdfs/  →  store it in ChromaDB
#
# Run this script ONCE before starting the API server so that
# the vector store is populated and ready to answer questions.
#
# How to run:
#   python setup_documents.py
#
# What it does (in order):
#   1. Scans data/pdfs/ for .pdf files
#   2. For each PDF → calls PDFProcessor  (see src/preprocessing.py)
#   3. Takes the chunks → calls VectorStore (see app/rag/vectorstore.py)
#   4. Prints a short summary
#
# This file contains ZERO chunking logic and ZERO embedding logic.
# It only coordinates the two modules above.
# =============================================================

import os
import sys
from pathlib import Path

# Make sure Python can find the project's own modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ingest_all_pdfs(pdf_dir: str = "data/pdfs"):
    # Import here so errors appear with a clean message
    from src.preprocessing import PDFProcessor
    from app.rag.vectorstore import create_vector_store

    processor    = PDFProcessor()
    vector_store = create_vector_store()

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"❌  No PDF files found in {pdf_dir}/")
        print("    → Put your PDF manuals in data/pdfs/ and run this script again.")
        return

    print(f"\n📂 Found {len(pdf_files)} PDF(s) in {pdf_dir}/")
    print("─" * 50)

    for pdf_path in pdf_files:
        # Step 1: PDF → chunks  (all logic lives in src/preprocessing.py)
        chunks = processor.process_pdf_file(str(pdf_path))

        # Step 2: chunks → ChromaDB  (all logic lives in app/rag/vectorstore.py)
        vector_store.add_documents(chunks)
        print(f"   ✓ Stored {len(chunks)} chunks for {pdf_path.name}")

    # Final summary
    stats = vector_store.get_collection_stats()
    print("─" * 50)
    print(f"📊 Vector store now contains {stats['total_documents']} documents.")
    print(f"   Collection : {stats['collection_name']}")
    print(f"   Model      : {stats['embedding_model']}")
    print("\n🎉 Done — you can now start the API server and query the RAG system.")


if __name__ == "__main__":
    ingest_all_pdfs()