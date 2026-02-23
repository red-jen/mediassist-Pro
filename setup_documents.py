# setup_documents.py  —  run once to load PDFs into ChromaDB
# Usage: python setup_documents.py
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing   import PDFProcessor
from app.rag.vectorstore import create_vector_store

processor    = PDFProcessor()
vector_store = create_vector_store()

for pdf in Path("data/pdfs").glob("*.pdf"):
    chunks = processor.process_pdf_file(str(pdf))
    vector_store.add_documents(chunks)

print(f"Done — {vector_store.get_collection_stats()['total_documents']} chunks in ChromaDB.")


if __name__ == "__main__":
    ingest_all_pdfs()