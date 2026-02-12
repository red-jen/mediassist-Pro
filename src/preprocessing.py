import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    """Simple PDF processing and chunking."""
    
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 800))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        
        # Why Recursive? It respects document structure
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    "filename": Path(pdf_path).name,
                    "total_pages": len(pdf_reader.pages),
                    "processed_at": datetime.now().isoformat(),
                }
                
                # Extract text page by page
                text_by_page = []
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_by_page.append({
                        "page_number": page_num + 1,
                        "text": page_text
                    })
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                return {
                    "metadata": metadata,
                    "full_text": full_text,
                    "pages": text_by_page
                }
                
        except Exception as e:
            raise Exception(f"Error processing {pdf_path}: {str(e)}")
    
    def create_chunks(self, pdf_data: Dict[str, Any]) -> List[Document]:
        """Create semantic chunks from PDF text."""
        documents = []
        
        # Option 1: Chunk the full document (maintains context across pages)
        full_text = pdf_data["full_text"]
        chunks = self.text_splitter.split_text(full_text)
        
        for i, chunk_text in enumerate(chunks):
            # Find which page(s) this chunk likely comes from
            page_info = self._find_chunk_pages(chunk_text, pdf_data["pages"])
            
            # Create Document with metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "filename": pdf_data["metadata"]["filename"],
                    "chunk_id": f"chunk_{i}",
                    "chunk_size": len(chunk_text),
                    "total_pages": pdf_data["metadata"]["total_pages"],
                    "likely_pages": page_info,
                    "processed_at": pdf_data["metadata"]["processed_at"]
                }
            )
            documents.append(doc)
        
        return documents
    
    def _find_chunk_pages(self, chunk_text: str, pages: List[Dict]) -> List[int]:
        """Find which pages contain parts of this chunk."""
        matching_pages = []
        
        # Simple heuristic: if chunk contains text from a page
        for page in pages:
            if any(word in page["text"] for word in chunk_text.split()[:10]):
                matching_pages.append(page["page_number"])
        
        return matching_pages[:3]  # Limit to 3 most likely pages
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """Complete processing pipeline for one PDF."""
        print(f"📄 Processing: {Path(pdf_path).name}")
        
        # Step 1: Extract text
        pdf_data = self.extract_text_from_pdf(pdf_path)
        print(f"   ✓ Extracted {pdf_data['metadata']['total_pages']} pages")
        
        # Step 2: Create chunks
        chunks = self.create_chunks(pdf_data)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Step 3: Save processed data (for inspection)
        self._save_processed_data(pdf_data, chunks)
        
        return chunks
    
    def _save_processed_data(self, pdf_data: Dict, chunks: List[Document]):
        """Save processed data for inspection."""
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(pdf_data["metadata"]["filename"]).stem
        
        # Save chunk summary
        chunk_summary = {
            "metadata": pdf_data["metadata"],
            "chunks": [
                {
                    "chunk_id": doc.metadata["chunk_id"],
                    "content_preview": doc.page_content[:100] + "...",
                    "size": doc.metadata["chunk_size"],
                    "likely_pages": doc.metadata["likely_pages"]
                }
                for doc in chunks
            ]
        }
        
        with open(output_dir / f"{filename}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_summary, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Saved processing summary to data/processed/{filename}_chunks.json")

# Convenience function for testing
def process_single_pdf(pdf_path: str) -> List[Document]:
    """Quick function to process one PDF and inspect results."""
    processor = PDFProcessor()
    return processor.process_pdf_file(pdf_path)