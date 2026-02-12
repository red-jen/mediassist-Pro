# ============================================================
# MULTI-STAGE DOCKERFILE — MediAssist Pro
# ============================================================
# Stage 1: builder — installs dependencies in a clean venv
# Stage 2: runtime — only what is needed to run the app
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System tools needed only during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Metadata
LABEL maintainer="AI-Cognitech <support@ai-cognitech.com>"
LABEL description="MediAssist Pro — Biomedical RAG System"
LABEL version="1.0.0"

WORKDIR /app

# System dependencies (runtime only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder (no build tools in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app/       ./app/
COPY src/       ./src/
COPY data/      ./data/
COPY .env       ./.env

# Create necessary runtime directories
RUN mkdir -p data/chroma_db data/uploads data/processed data/pdfs

# Non-root user for security
RUN groupadd -r mediassist && useradd -r -g mediassist mediassist
RUN chown -R mediassist:mediassist /app
USER mediassist

# Healthcheck: poll the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Entrypoint: production uvicorn (no reload, 2 workers)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
