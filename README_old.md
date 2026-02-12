# MediAssist Pro - Phase 1: Preprocessing

## What We're Learning

**Goal**: Understand how PDF preprocessing and chunking works before building the full RAG system.

### Key Concepts:
1. **Why Recursive Chunking?** Preserves document structure (paragraphs → sentences → words)
2. **Why Overlap?** Prevents context loss at chunk boundaries  
3. **Why Metadata?** Helps with retrieval and source attribution

## Quick Start

1. **Setup**:
```bash
pip install -r requirements.txt
```

2. **Add your PDF**: Place your biomedical manual in `data/pdfs/`

3. **Test preprocessing**:
```bash
cd tests
python test_preprocessing.py
```

4. **Inspect results**: Check `data/processed/` for chunk summaries

## Project Structure
```
MediAssist Pro/
├── data/
│   ├── pdfs/              # Your PDF files here
│   └── processed/         # Chunked data output
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # PDF → chunks
│   └── utils.py          # Helper functions
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── .env
└── README.md
```

## Current Status: ✅ Phase 1 - Preprocessing
- [x] PDF text extraction
- [x] Recursive chunking with overlap
- [x] Metadata enrichment
- [x] Processing inspection tools

## Understanding the Code

### Why RecursiveCharacterTextSplitter?
```python
separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
```
This tries to split on:
1. Paragraphs first (\n\n)
2. Then lines (\n)  
3. Then sentences (., !, ?)
4. Finally words and characters

This preserves semantic meaning better than fixed-size chunks.

### Why Chunk Overlap?
```python
chunk_overlap=100  # characters
```
Prevents losing context when important information spans chunk boundaries.

### Why Page Tracking?
```python
"likely_pages": [1, 2]
```
Helps users trace answers back to original document pages.

## Next: Phase 2 - Embeddings & Vector Store

Once you understand chunking, we'll add:
- Sentence transformers for embeddings
- ChromaDB for vector storage
- Similarity search capabilities