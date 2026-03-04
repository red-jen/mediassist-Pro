import time
from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, Gauge

from backend.models.query import Query
from backend.services.mlflow_service import log_rag_query
from backend.services.deepeval_service import compute_rag_metrics
from rag_engine import ask, index_pdf as engine_index_pdf

rag_queries_total = Counter(
    'rag_queries_total',
    'Total number of RAG queries processed'
)

rag_errors_total = Counter(
    'rag_errors_total',
    'Total number of RAG query errors'
)

rag_latency_seconds = Histogram(
    'rag_latency_seconds',
    'RAG query latency in seconds',
    buckets=[0.5, 1, 2, 5, 10, 30, 60]                     
)

rag_answer_relevance = Gauge(
    'rag_answer_relevance',
    'Latest answer relevance score from DeepEval (0-1)'
)

rag_faithfulness = Gauge(
    'rag_faithfulness',
    'Latest faithfulness score from DeepEval (0-1)'
)

rag_contextual_relevance = Gauge(
    'rag_contextual_relevance',
    'Latest contextual relevance score from DeepEval (0-1)'
)

def ask_question(query: str, chat_history: list, user_id: int, db: Session) -> dict:
    """
    Run the full RAG pipeline, save the result to PostgreSQL, and log to MLflow.

    Args:
        query        : user's question
        chat_history : list of previous {"question", "answer"} dicts
        user_id      : ID of the authenticated user making the request
        db           : SQLAlchemy session (injected by FastAPI)

    Returns:
        {"answer": str, "sources": list}
    """

    start_time = time.time()

    try:
        result = ask(query, chat_history)
        rag_queries_total.inc()                                       
    except Exception as e:
        rag_errors_total.inc()                                        
        raise e

    response_time = time.time() - start_time               
    rag_latency_seconds.observe(response_time)                              

    db_query = Query(
        user_id  = user_id,
        query    = query,
        reponse  = result["answer"]
    )
    db.add(db_query)
    db.commit()

    deepeval_scores = None
    try:
        deepeval_scores = compute_rag_metrics(
            query    = query,
            answer   = result["answer"],
            contexts = result.get("contexts", []),
        )
        print(f"[DeepEval] Scores: {deepeval_scores}")

        if deepeval_scores:
            if "answer_relevancy" in deepeval_scores:
                rag_answer_relevance.set(deepeval_scores["answer_relevancy"])
            if "faithfulness" in deepeval_scores:
                rag_faithfulness.set(deepeval_scores["faithfulness"])
            if "contextual_relevancy" in deepeval_scores:
                rag_contextual_relevance.set(deepeval_scores["contextual_relevancy"])

    except Exception as e:
        print(f"[DeepEval] Evaluation failed (non-critical): {e}")

    try:
        log_rag_query(
            query            = query,
            answer           = result["answer"],
            sources          = result.get("sources", []),
            response_time    = response_time,
            user_id          = user_id,
            deepeval_scores  = deepeval_scores,
        )
    except Exception as e:
        print(f"[MLflow] Logging failed (non-critical): {e}")

    return result

def index_document(pdf_path: str) -> dict:
    """
    Index a PDF into ChromaDB and BM25.
    Delegates to rag_engine.index_pdf().
    """
    return engine_index_pdf(pdf_path)
