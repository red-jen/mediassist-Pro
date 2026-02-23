# src/preprocessing.py  —  PDF  →  text chunks
import os
from pathlib import Path
from typing import List

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class PDFProcessor:
    """PDF file  →  list of Document chunks ready for the vector store."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        filename = Path(pdf_path).name

        # 1. Read all pages into one string
        full_text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                full_text += f"\n--- Page {i} ---\n{page.extract_text() or ''}"

        # 2. Split into overlapping chunks, attach source metadata
        raw = self.splitter.split_text(full_text)
        chunks = [
            Document(
                page_content=text,
                metadata={"filename": filename, "chunk_id": f"chunk_{i}"},
            )
            for i, text in enumerate(raw)
        ]

        print(f"  {filename}  →  {len(chunks)} chunks")
        return chunks