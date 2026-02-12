# MediAssist Pro - Biomedical Equipment Technical Support RAG System

> **Learning-Focused Implementation**: This project is designed for educational purposes to understand RAG system architecture, implementation, and deployment.

## 📋 Project Overview

MediAssist Pro is an intelligent document retrieval and question-answering system designed for biomedical laboratory equipment support. It helps technicians quickly find accurate, actionable answers from technical manuals and documentation.

## 🎯 Learning Objectives

This implementation teaches:

1. **RAG Architecture**: Complete Retrieval-Augmented Generation pipeline
2. **Document Processing**: PDF parsing, chunking, and metadata extraction  
3. **Vector Storage**: Embedding generation and similarity search
4. **API Development**: FastAPI with authentication and structured responses
5. **System Integration**: Database, security, error handling, and deployment

## 🏗️ System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PDF Docs  │───▶│  Chunking   │───▶│ Embeddings  │───▶│ Vector DB   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  FastAPI    │◀───│     LLM     │◀───│  Retriever  │◀─────────────┘
│  Frontend   │    │  Response   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 📁 Project Structure

```
MediAssist Pro/
├── app/                          # Main application package
│   ├── main.py                  # FastAPI application entry point
│   ├── core/                    # Core application components
│   │   ├── config.py           # Configuration management
│   │   ├── security.py         # Authentication & authorization
│   │   └── exceptions.py       # Error handling
│   ├── db/                     # Database layer
│   │   ├── models.py           # SQLAlchemy models
│   │   └── session.py          # Database session management
│   ├── rag/                    # RAG system components
│   │   ├── embeddings.py       # Text-to-vector conversion
│   │   ├── vectorstore.py      # ChromaDB integration
│   │   ├── retrieval.py        # Smart document retrieval
│   │   └── chain.py            # Complete RAG pipeline
│   └── api/                    # API routes
│       └── routes/
│           ├── auth.py         # Authentication endpoints
│           └── query.py        # RAG query endpoints
├── src/                        # Legacy preprocessing (integrated)
│   ├── preprocessing.py        # PDF processing utilities
│   └── utils.py               # Helper functions
├── data/                       # Data storage
│   ├── chroma_db/             # Vector database persistence
│   └── processed/             # Processing outputs
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
├── .env                       # Environment configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### 1. Installation

```bash
# Navigate to project directory
cd "MediAssist Pro"

# Create virtual environment
python -m venv venv
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The `.env` file is already configured with good defaults for learning.

### 3. Add Your Documents

```bash
# Your PDF is already in the data directory
# The system will process it automatically on startup
```

### 4. Run the Application

```bash
# Start the development server
python -m app.main
```

### 5. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Info**: http://localhost:8000/

## 🔑 Authentication

### Default Users (Development)

The system creates default users automatically:

```
Admin User: admin / (will create default password)
Tech User: tech1 / (will create default password)
```

### Getting Access Token

1. Go to http://localhost:8000/docs
2. Click "Authorize" and use the login endpoint
3. Copy the returned access token

## 🤖 RAG System Components (Key Learning Points)

### 1. Document Processing (Chunking)

**Strategy**: Recursive Character Text Splitter
- **Why**: Preserves semantic boundaries (paragraphs → sentences → words)
- **Settings**: 800 chars with 100 char overlap (12.5%)
- **Learning**: Understand how chunking affects retrieval quality

### 2. Embeddings

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Why**: Balanced speed vs. quality, runs locally, no API costs
- **Dimensions**: 384 (efficient for storage and search)
- **Learning**: See how text becomes searchable vectors

### 3. Vector Storage

**Database**: ChromaDB
- **Why**: Lightweight, persistent, excellent for learning
- **Learning**: Understand similarity search and metadata filtering

### 4. Retrieval Strategy

**Approach**: Hybrid (Query Expansion + Reranking)
- **Query Expansion**: Generate alternative phrasings
- **Reranking**: Improve relevance of results
- **Learning**: See how different strategies affect answer quality

### 5. Response Generation

**Model**: Ollama (llama3.2) or Mock LLM for testing
- **Prompt Engineering**: Grounded, anti-hallucination prompts
- **Learning**: Understand how prompts prevent hallucinations

## 📝 API Usage Examples

Once running, try these in the API documentation at `/docs`:

### 1. Login
```json
{
  "username": "admin",
  "password": "your_password"
}
```

### 2. Ask a Question
```json
{
  "question": "How do I maintain laboratory equipment?",
  "retrieval_strategy": "hybrid"
}
```

### 3. Check Query History
Use the `/api/v1/query/history` endpoint

## 📊 System Features (What You'll Learn)

### ✅ Authentication & Authorization
- JWT-based authentication
- Role-based access control 
- Password validation and hashing

### ✅ RAG Capabilities
- Semantic document search
- Multi-strategy retrieval
- Source attribution and confidence scoring
- Query expansion and reranking

### ✅ Data Management
- PDF document processing
- Metadata extraction and storage
- Vector embedding persistence
- Query logging and analytics

### ✅ API & Integration
- RESTful API with OpenAPI documentation
- Structured error handling
- Request validation
- Health monitoring

## 🧪 Testing the System

### Test Individual Components

```bash
# Test preprocessing
python src/preprocessing.py

# Test embeddings
python app/rag/embeddings.py

# Test vector store
python app/rag/vectorstore.py

# Test complete RAG chain
python app/rag/chain.py
```

### Test via API

1. Start the server: `python -m app.main`
2. Go to http://localhost:8000/docs
3. Try the authentication and query endpoints

## 🎓 Learning Path

### Beginner: Understand Components
1. **Start with preprocessing**: See how PDFs become chunks
2. **Explore embeddings**: Understand text-to-vector conversion
3. **Try vector storage**: Learn similarity search
4. **Test retrieval**: Compare different strategies
5. **Use the API**: See the complete flow

### Intermediate: Modify and Extend
1. **Adjust chunking**: Try different sizes and overlaps
2. **Change embeddings**: Test different models
3. **Tune retrieval**: Modify similarity thresholds
4. **Custom prompts**: Improve response generation

### Advanced: System Design
1. **Add new endpoints**: Extend the API
2. **Improve security**: Add rate limiting
3. **Add monitoring**: Track system performance
4. **Scale components**: Understand bottlenecks

## 🐛 Troubleshooting

### Common Issues

**"No LLM available"**
- The system uses a mock LLM by default for learning
- Install Ollama for real LLM: `ollama pull llama3.2`
- Or configure OpenAI API key in `.env`

**"Database errors"**
- The system creates SQLite database automatically
- Check file permissions in project directory

**"No documents found"**
- Ensure your PDF is in the data directory
- Check startup logs for processing errors

## 🤝 Educational Value

This project teaches:

1. **RAG Architecture**: Complete end-to-end implementation
2. **API Design**: RESTful services with FastAPI
3. **Database Integration**: SQLAlchemy ORM patterns
4. **Security**: JWT authentication and authorization
5. **Error Handling**: Graceful error management
6. **Testing**: Component and integration testing
7. **Configuration**: Environment-based settings
8. **Documentation**: API documentation and code comments

## 📚 Next Steps

After understanding this implementation:

1. **Extend functionality**: Add document upload, user management
2. **Improve performance**: Implement caching, optimize queries
3. **Add monitoring**: Metrics collection and alerting
4. **Frontend integration**: Build a web interface
5. **Production deployment**: Docker, load balancing, scaling

---

**Remember**: This is a learning project! Focus on understanding each component and how they work together to create an intelligent document search system.