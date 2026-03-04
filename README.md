# 🩺 MediAssist Pro

**RAG-powered medical assistant for healthcare professionals**

MediAssist Pro is an AI solution that helps medical professionals quickly access information from biomedical device manuals and clinical protocols using Retrieval-Augmented Generation (RAG).

---

## 📋 Features

- **RAG Pipeline**: Hybrid retrieval (ChromaDB + BM25) with cross-encoder reranking
- **Query Rewriting**: Automatic query expansion for better retrieval
- **JWT Authentication**: Secure user registration and login
- **MLflow Tracking**: Logs all RAG parameters, responses, and quality metrics
- **DeepEval Metrics**: Neural evaluation (Answer Relevancy, Faithfulness, Contextual Relevancy)
- **Prometheus + Grafana**: Real-time monitoring dashboards
- **Docker**: Fully containerized deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────┐                              ┌─────────────────┐        │
│    │  Streamlit  │ ◄──── HTTP/REST ────────►   │   FastAPI       │        │
│    │    :8501    │                              │     :8000       │        │
│    └─────────────┘                              └────────┬────────┘        │
│                                                          │                  │
├──────────────────────────────────────────────────────────┼──────────────────┤
│                           BACKEND SERVICES               │                  │
├──────────────────────────────────────────────────────────┼──────────────────┤
│                                                          │                  │
│    ┌──────────────────────────────────────────────────────┘                 │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Auth Service  │    │   RAG Service   │    │  MLflow Service │         │
│  │  (JWT/bcrypt)   │    │                 │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   PostgreSQL    │    │   RAG Engine    │    │     MLflow      │         │
│  │     :5432       │    │                 │    │     :5000       │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                        RAG PIPELINE                                         │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │                                          │
│    ┌─────────────────────────────┴─────────────────────────────┐           │
│    │                                                           │           │
│    ▼                                                           ▼           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │    ChromaDB     │    │      BM25       │    │  Cross-Encoder  │         │
│  │  (Dense Index)  │    │ (Sparse Index)  │    │   (Reranker)    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┴──────────────────────┘                   │
│                                  │                                          │
│                                  ▼                                          │
│                        ┌─────────────────┐                                  │
│                        │  Ollama (LLM)   │                                  │
│                        │    Mistral      │                                  │
│                        │    :11434       │                                  │
│                        └─────────────────┘                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MONITORING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────┐                      ┌─────────────────┐            │
│    │   Prometheus    │ ────── scrape ─────► │    FastAPI      │            │
│    │     :9090       │                      │   /metrics      │            │
│    └────────┬────────┘                      └─────────────────┘            │
│             │                                                               │
│             ▼                                                               │
│    ┌─────────────────┐                                                      │
│    │    Grafana      │                                                      │
│    │     :3000       │                                                      │
│    └─────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (Windows/Mac) or Docker + Docker Compose (Linux)
- **Ollama** running on host with `mistral` model:
  ```bash
  ollama pull mistral
  ollama serve
  ```

### 1. Clone & Configure

```bash
git clone https://github.com/your-username/mediassist-pro.git
cd mediassist-pro

# Copy environment template
cp .env.example .env

# Edit .env and set your SECRET_KEY
```

### 2. Start with Docker

```bash
docker compose up --build -d
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit UI | http://localhost:8501 | Register new account |
| FastAPI Docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| MLflow | http://localhost:5000 | — |

---

## 📁 Project Structure

```
mediassist-pro/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Pydantic settings
│   ├── database.py          # SQLAlchemy engine & session
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── user.py
│   │   └── query.py
│   ├── schemas/             # Pydantic request/response schemas
│   │   ├── user.py
│   │   └── rag.py
│   ├── routers/             # API route handlers
│   │   ├── auth.py          # /auth/register, /auth/login
│   │   └── rag.py           # /rag/ask, /rag/upload-pdf
│   └── services/            # Business logic
│       ├── auth_service.py  # JWT, password hashing
│       ├── rag_service.py   # RAG orchestration
│       ├── mlflow_service.py
│       └── deepeval_service.py
├── rag_engine.py            # Core RAG pipeline
├── app.py                   # Streamlit frontend
├── tests/                   # Unit tests
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_auth.py
│   └── test_rag.py
├── grafana/                 # Grafana provisioning
│   ├── dashboards/
│   └── provisioning/
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── .env.example
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key | (required) |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://postgres:postgres@localhost:5432/mediassist` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry | `30` |

### RAG Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Chunking | RecursiveCharacterTextSplitter, size=1000, overlap=150 |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| Dense Retrieval | ChromaDB, k=8 |
| Sparse Retrieval | BM25, k=8 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Final k | 3 (after reranking) |
| LLM | Mistral via Ollama, temperature=0 |

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create new user |
| POST | `/auth/login` | Get JWT token |

### RAG

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/rag/ask` | Ask a question | Required |
| POST | `/rag/upload-pdf` | Index a PDF document | Required |

### Health & Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

---

## 🧪 Testing

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Run tests
pytest -v

# Run with coverage
pytest --cov=backend --cov-report=html
```

---

## 📊 Monitoring

### Grafana Dashboard

The pre-configured dashboard includes:
- **Total RAG Queries** — Counter
- **Total RAG Errors** — Counter
- **Avg RAG Latency** — Histogram
- **Answer Relevance** — DeepEval gauge
- **Faithfulness** — DeepEval gauge
- **HTTP Requests by Method** — Time series
- **CPU/Memory Usage** — Process metrics

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_queries_total` | Counter | Total RAG queries |
| `rag_errors_total` | Counter | Total errors |
| `rag_latency_seconds` | Histogram | Query latency |
| `rag_answer_relevance` | Gauge | DeepEval score |
| `rag_faithfulness` | Gauge | DeepEval score |
| `http_requests_total` | Counter | HTTP requests |

---

## 🔄 CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Test** — Run pytest
2. **Build** — Build Docker image
3. **Push** — Push to Docker Hub (requires secrets)

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `SECRET_KEY` | JWT signing key |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password/token |

---

## 📚 Documentation

- [DeepEval Metrics](docs/deepeval.md) — Explanation of RAG quality metrics
- [Tests Guide](docs/tests.md) — How to write and run tests

---

## 🛠️ Development

### Local Setup (without Docker)

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (local or Docker)
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16-alpine

# Start Ollama
ollama serve

# Run FastAPI
uvicorn backend.main:app --reload

# Run Streamlit (separate terminal)
streamlit run app.py
```

---

## 📄 License

MIT License

---

## 👤 Author

ProtoCare AI Team — Jury Blanc Project (Feb 2026)
