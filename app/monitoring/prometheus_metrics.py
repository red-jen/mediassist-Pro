"""
PROMETHEUS METRICS
==================

Exposes application-level metrics on the /metrics endpoint using
prometheus_client.  FastAPI mounts this via starlette-prometheus.

Metric types used:
  Counter   – monotonically increasing counts  (requests, errors)
  Histogram – latency distributions (bucketed)
  Gauge     – snapshot values (documents in store, active users)

All metrics are prefixed with  mediassist_  to avoid collisions in
a shared Prometheus/Grafana deployment.

Usage (in main.py):
    from app.monitoring.prometheus_metrics import RAGMetrics, setup_prometheus
    setup_prometheus(app)    # mounts /metrics + adds middleware
    metrics = RAGMetrics()
    metrics.record_query(latency_ms=420, confidence=0.82, success=True)
"""

import time
from functools import wraps
from typing import Callable

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        REGISTRY,
        make_asgi_app,
    )
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Metric Definitions  (module-level singletons)
# ─────────────────────────────────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    # ── Request counters ──────────────────────────────────────────
    HTTP_REQUESTS_TOTAL = Counter(
        "mediassist_http_requests_total",
        "Total number of HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    # ── RAG query counters ────────────────────────────────────────
    RAG_QUERIES_TOTAL = Counter(
        "mediassist_rag_queries_total",
        "Total RAG queries processed",
        ["strategy", "status"],           # status: success | error | no_docs
    )

    RAG_ERRORS_TOTAL = Counter(
        "mediassist_rag_errors_total",
        "Total RAG errors",
        ["error_type"],
    )

    # ── Latency histograms ────────────────────────────────────────
    RAG_LATENCY_SECONDS = Histogram(
        "mediassist_rag_latency_seconds",
        "End-to-end RAG query latency in seconds",
        ["strategy"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    HTTP_LATENCY_SECONDS = Histogram(
        "mediassist_http_latency_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # ── Quality gauges (updated after each query) ─────────────────
    RAG_CONFIDENCE_GAUGE = Gauge(
        "mediassist_rag_confidence_last",
        "Confidence score of the last RAG response",
    )

    RAG_FAITHFULNESS_GAUGE = Gauge(
        "mediassist_rag_faithfulness_last",
        "Faithfulness score of the last RAG response",
    )

    RAG_ANSWER_RELEVANCE_GAUGE = Gauge(
        "mediassist_rag_answer_relevance_last",
        "Answer relevance score of the last RAG response",
    )

    RAG_PRECISION_K_GAUGE = Gauge(
        "mediassist_rag_precision_k_last",
        "Precision@k of the last retrieval",
    )

    RAG_RECALL_K_GAUGE = Gauge(
        "mediassist_rag_recall_k_last",
        "Recall@k of the last retrieval",
    )

    # ── Infrastructure gauges ─────────────────────────────────────
    DOCS_IN_STORE_GAUGE = Gauge(
        "mediassist_docs_in_vectorstore",
        "Number of document chunks currently in the vector store",
    )

    ACTIVE_USERS_GAUGE = Gauge(
        "mediassist_active_users",
        "Number of active (non-disabled) users in the database",
    )

    # ── Build info (useful label for dashboards) ──────────────────
    APP_INFO = Info(
        "mediassist_app",
        "MediAssist Pro application info",
    )


# ─────────────────────────────────────────────────────────────────
# RAGMetrics class — convenience wrapper
# ─────────────────────────────────────────────────────────────────

class RAGMetrics:
    """
    Thin wrapper around the Prometheus metric objects above.
    Import this class everywhere you want to record observations.
    """

    def __init__(self):
        self.available = PROMETHEUS_AVAILABLE
        if not self.available:
            print("⚠️  prometheus_client not installed — metrics disabled")

    # ── App info (call once at startup) ──────────────────────────
    def set_app_info(self, version: str, env: str, embedding_model: str, llm_model: str):
        if not self.available:
            return
        APP_INFO.info({
            "version": version,
            "environment": env,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
        })

    # ── Record a completed RAG query ──────────────────────────────
    def record_query(
        self,
        latency_ms: float,
        confidence: float,
        strategy: str = "hybrid",
        success: bool = True,
        deepeval_scores: dict = None,
    ):
        if not self.available:
            return

        status = "success" if success else "error"
        latency_s = latency_ms / 1000.0

        RAG_QUERIES_TOTAL.labels(strategy=strategy, status=status).inc()
        RAG_LATENCY_SECONDS.labels(strategy=strategy).observe(latency_s)
        RAG_CONFIDENCE_GAUGE.set(confidence)

        if deepeval_scores:
            RAG_FAITHFULNESS_GAUGE.set(deepeval_scores.get("faithfulness", 0.0))
            RAG_ANSWER_RELEVANCE_GAUGE.set(deepeval_scores.get("answer_relevance", 0.0))
            RAG_PRECISION_K_GAUGE.set(deepeval_scores.get("precision_k", 0.0))
            RAG_RECALL_K_GAUGE.set(deepeval_scores.get("recall_k", 0.0))

    # ── Record an error ───────────────────────────────────────────
    def record_error(self, error_type: str = "unknown"):
        if not self.available:
            return
        RAG_ERRORS_TOTAL.labels(error_type=error_type).inc()

    # ── Update infrastructure gauges ─────────────────────────────
    def update_infrastructure(self, docs_in_store: int, active_users: int):
        if not self.available:
            return
        DOCS_IN_STORE_GAUGE.set(docs_in_store)
        ACTIVE_USERS_GAUGE.set(active_users)

    # ── HTTP request shortcut ─────────────────────────────────────
    def record_http(self, method: str, endpoint: str, status_code: int, latency_s: float):
        if not self.available:
            return
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        HTTP_LATENCY_SECONDS.labels(method=method, endpoint=endpoint).observe(latency_s)


# ─────────────────────────────────────────────────────────────────
# FastAPI integration
# ─────────────────────────────────────────────────────────────────

def setup_prometheus(app) -> RAGMetrics:
    """
    Mount the /metrics endpoint and add a latency-tracking middleware.
    Call this ONCE in app/main.py during startup.

    Returns a RAGMetrics instance ready to use.
    """
    if not PROMETHEUS_AVAILABLE:
        print("⚠️  Prometheus not available — /metrics endpoint not mounted")
        return RAGMetrics()

    # Mount the ASGI metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Middleware to auto-record HTTP latency
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable):
        start = time.time()
        response = await call_next(request)
        latency = time.time() - start

        endpoint = request.url.path
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        HTTP_LATENCY_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(latency)

        return response

    print("✅ Prometheus /metrics endpoint mounted")
    return RAGMetrics()
