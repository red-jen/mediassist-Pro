"""
MLFLOW RAG TRACKER
==================

Logs every requirement from Part 2 of the project brief:

1. RAG configuration:     chunking params, embedding model, retrieval config
2. LLM hyperparameters:   template, temperature, model, max_tokens, top_p, top_k
3. Responses & contexts:  logged as artifacts per query
4. RAG metrics:           Answer Relevance, Faithfulness, Precision@k, Recall@k
5. LangChain pipeline:    registered in MLflow Model Registry

Usage:
    tracker = RAGMLflowTracker()
    with tracker.start_rag_run(config) as run:
        tracker.log_query(question, context, answer, metrics)
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

import mlflow
import mlflow.langchain
from dotenv import load_dotenv

load_dotenv()


class RAGMLflowTracker:
    """
    Central MLflow tracker for the MediAssist Pro RAG system.

    WHAT IS LOGGED:
    ─────────────────────────────────────────────────────────────
    params  │ chunking, embedding, retrieval, LLM hyper-params
    metrics │ faithfulness, answer_relevance, precision_k, recall_k,
            │ confidence, latency_ms, retrieved_chunks
    tags    │ model, environment, query_id
    artifacts│ full prompt context + LLM response (JSON per query)
    model   │ LangChain RAG pipeline (MLflow Model Registry)
    ─────────────────────────────────────────────────────────────
    """

    EXPERIMENT_NAME = "MediAssist-RAG"

    def __init__(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Create or reuse the experiment
        experiment = mlflow.get_experiment_by_name(self.EXPERIMENT_NAME)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.EXPERIMENT_NAME,
                tags={
                    "project": "MediAssist Pro",
                    "domain": "biomedical-rag",
                    "version": os.getenv("APP_VERSION", "1.0.0"),
                },
            )
        else:
            self.experiment_id = experiment.experiment_id

        print(f"📊 MLflow tracker ready — experiment: {self.EXPERIMENT_NAME}")
        print(f"   Tracking URI: {tracking_uri}")

    # ─────────────────────────────────────────────────────────────
    # 1.  LOG RAG CONFIGURATION
    # ─────────────────────────────────────────────────────────────
    def log_rag_config(self, run: mlflow.ActiveRun, config: Dict[str, Any]) -> None:
        """
        Log every RAG configuration parameter required by the project brief.

        Expected keys in `config`:
          chunking   – chunk_size, chunk_overlap, strategy
          embedding  – model, dimension, normalize
          retrieval  – similarity_metric, top_k, reranking, strategy
          llm        – model, temperature, max_tokens, top_p, top_k,
                       prompt_template_hash, prompt_length
        """
        params: Dict[str, Any] = {}

        # --- Chunking ---
        chunking = config.get("chunking", {})
        params["chunk_size"] = chunking.get("chunk_size", 800)
        params["chunk_overlap"] = chunking.get("chunk_overlap", 100)
        params["chunk_strategy"] = chunking.get("strategy", "recursive_character")

        # --- Embedding ---
        embedding = config.get("embedding", {})
        params["embedding_model"] = embedding.get("model", "all-MiniLM-L6-v2")
        params["embedding_dimension"] = embedding.get("dimension", 384)
        params["embedding_normalize"] = embedding.get("normalize", True)

        # --- Retrieval ---
        retrieval = config.get("retrieval", {})
        params["similarity_metric"] = retrieval.get("similarity_metric", "cosine")
        params["retrieval_top_k"] = retrieval.get("top_k", 5)
        params["reranking_enabled"] = retrieval.get("reranking", True)
        params["retrieval_strategy"] = retrieval.get("strategy", "hybrid")
        params["similarity_threshold"] = retrieval.get("similarity_threshold", 0.1)

        # --- LLM ---
        llm = config.get("llm", {})
        params["llm_model"] = llm.get("model", "llama3.2")
        params["llm_temperature"] = llm.get("temperature", 0.1)
        params["llm_max_tokens"] = llm.get("max_tokens", 2048)
        params["llm_top_p"] = llm.get("top_p", 0.9)
        params["llm_top_k"] = llm.get("top_k", 40)
        params["prompt_template_length"] = llm.get("prompt_length", 0)
        params["prompt_template_hash"] = llm.get("prompt_hash", "")

        mlflow.log_params(params)
        print(f"   ✅ MLflow: {len(params)} RAG config params logged")

    # ─────────────────────────────────────────────────────────────
    # 2.  LOG QUERY (response + context as artifact)
    # ─────────────────────────────────────────────────────────────
    def log_query(
        self,
        run: mlflow.ActiveRun,
        question: str,
        context: str,
        answer: str,
        sources: List[str],
        query_id: Optional[int] = None,
    ) -> None:
        """
        Log the full prompt context and LLM response as a JSON artifact.
        One artifact file per query, named  query_<id>_<timestamp>.json
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        artifact_name = f"query_{query_id or 'anon'}_{timestamp}.json"

        payload = {
            "query_id": query_id,
            "timestamp": timestamp,
            "question": question,
            "context_used": context,
            "answer": answer,
            "sources": sources,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, artifact_name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(path, artifact_path="queries")

        mlflow.set_tag("last_query_id", str(query_id))

    # ─────────────────────────────────────────────────────────────
    # 3.  LOG RAG METRICS (DeepEval scores + retrieval KPIs)
    # ─────────────────────────────────────────────────────────────
    def log_metrics(
        self,
        run: mlflow.ActiveRun,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log RAG quality metrics.

        Expected keys (all 0–1 floats):
          faithfulness      – answer faithful to retrieved context
          answer_relevance  – answer relevant to the question
          precision_k       – retrieved docs that are relevant  (Precision@k)
          recall_k          – relevant docs that were retrieved  (Recall@k)
          confidence        – system confidence score
          latency_ms        – end-to-end latency in milliseconds
          retrieved_chunks  – number of chunks used
        """
        clean_metrics = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(clean_metrics, step=step)

    # ─────────────────────────────────────────────────────────────
    # 4.  LOG LANGCHAIN PIPELINE MODEL
    # ─────────────────────────────────────────────────────────────
    def log_langchain_pipeline(
        self,
        run: mlflow.ActiveRun,
        rag_chain,
        registered_model_name: str = "MediAssist-RAGChain",
    ) -> None:
        """
        Log the LangChain RAG pipeline to MLflow Model Registry.

        WHY? Enables model versioning, stage transitions
        (Staging → Production), and rollback.
        """
        try:
            mlflow.langchain.log_model(
                lc_model=rag_chain,
                artifact_path="rag_pipeline",
                registered_model_name=registered_model_name,
            )
            print(f"   ✅ MLflow: LangChain pipeline registered as '{registered_model_name}'")
        except Exception as e:
            # Graceful degradation: log a note instead of crashing
            mlflow.set_tag("pipeline_logging_error", str(e))
            print(f"   ⚠️  MLflow: Could not log LangChain model ({e})")

    # ─────────────────────────────────────────────────────────────
    # 5.  CONTEXT MANAGER — wrap a single RAG run
    # ─────────────────────────────────────────────────────────────
    def start_rag_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> mlflow.ActiveRun:
        """
        Context-manager that wraps a single MLflow run.

        Usage:
            with tracker.start_rag_run(config) as run:
                tracker.log_query(run, ...)
                tracker.log_metrics(run, {...})
        """
        run_name = run_name or f"rag-run-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags={
                "environment": os.getenv("APP_ENV", "development"),
                "llm_model": os.getenv("LLM_MODEL", "llama3.2"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            },
        )

        if config:
            self.log_rag_config(active_run, config)

        return active_run

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    # ─────────────────────────────────────────────────────────────
    # 6.  Build config dict from .env / settings (convenience)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def build_config_from_env() -> Dict[str, Any]:
        """
        Build the canonical RAG config dict from environment variables.
        Pass this to start_rag_run() so every run has consistent params.
        """
        import hashlib

        prompt_template = os.getenv("PROMPT_TEMPLATE", "")
        prompt_hash = hashlib.md5(prompt_template.encode()).hexdigest()[:8] if prompt_template else "default"

        return {
            "chunking": {
                "chunk_size": int(os.getenv("CHUNK_SIZE", 800)),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 100)),
                "strategy": "recursive_character",
            },
            "embedding": {
                "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "dimension": int(os.getenv("EMBEDDING_DIMENSION", 384)),
                "normalize": True,
            },
            "retrieval": {
                "similarity_metric": "cosine",
                "top_k": int(os.getenv("TOP_K", 5)),
                "reranking": True,
                "strategy": "hybrid",
                "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", 0.1)),
            },
            "llm": {
                "model": os.getenv("LLM_MODEL", "llama3.2"),
                "temperature": 0.1,
                "max_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40,
                "prompt_hash": prompt_hash,
                "prompt_length": len(prompt_template),
            },
        }
