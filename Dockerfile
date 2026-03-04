# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — MediAssist Pro FastAPI backend
# ─────────────────────────────────────────────────────────────────────────────
#
# Build stages:
#   1. python:3.12-slim base (minimal Debian image)
#   2. Install system packages needed to compile C extensions (psycopg2, etc.)
#   3. Install PyTorch CPU-only first (avoids pulling the 2 GB CUDA build)
#   4. Install the rest of our Python dependencies
#   5. Copy application source code
#   6. Launch uvicorn
#
# Note on Ollama:
#   Ollama runs on the HOST machine, not inside this container.
#   The container reaches it via OLLAMA_BASE_URL (set in docker-compose.yml).
#   On Windows/Mac Docker Desktop: http://host.docker.internal:11434
#   On Linux:                       extra_hosts host-gateway handles this.

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Working directory inside the container ────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc        : compile C extensions (psycopg2, chromadb)
# libpq-dev  : PostgreSQL client library headers (psycopg2 needs this)
# curl       : useful for healthcheck scripts
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch CPU-only ──────────────────────────────────────────────────────────
# Install torch BEFORE requirements.txt so pip doesn't pull the CUDA build.
# CPU-only wheel is ~200 MB vs ~2 GB for the CUDA build.
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────────────────────
# Copy everything; .dockerignore excludes .venv, __pycache__, .env, etc.
COPY . .

# ── Port the container listens on ────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health returns non-200.
# docker-compose "depends_on: condition: service_healthy" uses this.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# --host 0.0.0.0  : bind to all interfaces (not just localhost)
# --port 8000     : matches EXPOSE above
# --workers 1     : single worker (safe for in-memory RAG state like bm25_index)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
