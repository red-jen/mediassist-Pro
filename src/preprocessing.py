# =============================================================
# src/preprocessing.py
# ------------------------------------------------------------- 
# JOB: PDF file  →  list of text chunks (LangChain Documents)
#
# This is the ONLY place in the project that touches PDF files.
# It does three things, in order:
#   1. Open the PDF and read every page
#   2. Merge all pages into one text string
#   3. Split that text into overlapping chunks
#
# Nothing else lives here — no embeddings, no DB, no HTTP.
# =============================================================

import os
import json
from pathlib import Path
from typing import List
from datetime import datetime

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class PDFProcessor:
    """
    Converts a PDF file into a list of LangChain Document chunks.

    Each chunk is a small piece of the original text (~800 chars)
    with metadata so we can always trace it back to its source.

    Usage:
        processor = PDFProcessor()
        chunks = processor.process_pdf_file("data/pdfs/manual.pdf")
        # chunks is a list of Document objects ready for the vector store
    """

    def __init__(self):
        # Chunk size and overlap come from .env, with sensible defaults
        self.chunk_size    = int(os.getenv("CHUNK_SIZE",    800))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

        # RecursiveCharacterTextSplitter tries to split on paragraphs
        # first, then sentences, then words — so chunks stay readable
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    # ── Step 1 ────────────────────────────────────────────────
    def _read_pdf(self, pdf_path: str) -> str:
        """
        Open a PDF and return all its text as one big string.
        Each page is separated by a marker so we keep page context.
        """
        full_text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                full_text += f"\n--- Page {page_num} ---\n{page_text}"
        return full_text

    # ── Step 2 ────────────────────────────────────────────────
    def _split_into_chunks(self, text: str, filename: str) -> List[Document]:
        """
        Split the full text into overlapping chunks.
        Each chunk becomes a LangChain Document with metadata.
        """
        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "filename":     filename,
                    "chunk_id":     f"chunk_{i}",
                    "chunk_index":  i,
                    "total_chunks": len(raw_chunks),
                    "processed_at": datetime.now().isoformat(),
                }
            ))
        return chunks

    # ── Public API ────────────────────────────────────────────
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """
        Main method — the only one setup_documents.py needs to call.

        PDF path  →  list of Document chunks
        """
        filename = Path(pdf_path).name
        print(f"📄 Reading:   {filename}")

        text   = self._read_pdf(pdf_path)
        chunks = self._split_into_chunks(text, filename)

        print(f"   ✓ {len(chunks)} chunks created (size={self.chunk_size}, overlap={self.chunk_overlap})")
        self._save_debug_summary(filename, chunks)   # optional, helps inspection
        return chunks

    # ── Debug helper (optional) ───────────────────────────────
    def _save_debug_summary(self, filename: str, chunks: List[Document]):
        """
        Saves a JSON summary of chunks to data/processed/ so you can
        open it and inspect what was extracted without running the full app.
        """
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem

        summary = [
            {
                "chunk_id":  doc.metadata["chunk_id"],
                "preview":   doc.page_content[:120] + "...",
                "num_chars": len(doc.page_content),
            }
            for doc in chunks
        ]

        with open(out_dir / f"{stem}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"   ✓ Debug summary → data/processed/{stem}_chunks.json")