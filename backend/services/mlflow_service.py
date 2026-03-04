import json
import os
import tempfile
import mlflow

CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 150
CHUNKING_STRATEGY   = "RecursiveCharacterTextSplitter"

EMBEDDING_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM       = 384                                          
EMBEDDING_NORMALIZE = True                                                       

SIMILARITY_METRIC   = "cosine"
K_DENSE             = 8                                                 
K_SPARSE            = 8                                            
K_FINAL             = 3                                                   
RERANKER_MODEL      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RETRIEVER_TYPE      = "hybrid"                                    

LLM_MODEL           = "mistral"
LLM_TEMPERATURE     = 0                                  
LLM_MAX_TOKENS      = None                                    
LLM_TOP_P           = 1.0                       
LLM_TOP_K           = 40                        

PROMPT_TEMPLATE = (
    "medical_assistant_fr | Tu es un assistant technique biomédical. "
    "Réponds UNIQUEMENT avec les informations du CONTEXTE. "
    "Si l'information n'existe pas, dis: "
    "'Information non trouvée dans les documents.' "
    "Réponse courte, claire, actionnable."
)

EXPERIMENT_NAME = "mediassist-rag"

def _answer_relevance(query: str, answer: str) -> float:
    """
    Lexical relevance: fraction of unique query words present in the answer.
    Range: 0.0 (zero overlap) → 1.0 (every query word appears in the answer).

    Example:
        query = "what is the dosage for ibuprofen"
        answer = "The recommended dosage is 400mg"
        → overlap: {"dosage"} / 6 query words = 0.167
    """
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.0
    return round(len(q_words & a_words) / len(q_words), 3)

def _faithfulness(answer: str, sources: list) -> float:
    """
    Grounding check: was the answer backed by retrieved sources?
    1.0 = sources exist       (answer is likely grounded in documents)
    0.0 = no sources returned (answer may be hallucinated)
    """
    return 1.0 if sources else 0.0

def _precision_at_k(sources: list, k: int = K_FINAL) -> float:
    """
    Precision@k: of the k retrieved chunks, how many have a valid source ref?
    A valid source = source path is not empty / unknown.

    Example: 2 out of 3 chunks have a real source → precision = 0.667
    """
    if k == 0:
        return 0.0
    valid = sum(1 for s in sources if s.get("source", "").strip() not in ("", "unknown"))
    return round(valid / k, 3)

def _recall_at_k(sources: list, k: int = K_FINAL) -> float:
    """
    Recall@k: fraction of distinct source documents captured in the top-k chunks.
    Approximated as: unique_sources / k_final  (capped at 1.0)

    Example: 3 chunks all from the same PDF → recall = 1/3 = 0.333
             3 chunks from 3 different PDFs → recall = 3/3 = 1.0
    """
    if k == 0:
        return 0.0
    unique = len(set(s.get("source", "") for s in sources))
    return round(min(unique / k, 1.0), 3)

def log_rag_query(
    query:          str,
    answer:         str,
    sources:        list,
    response_time:  float,                                            
    user_id:        int,
    deepeval_scores: dict | None = None                                 
) -> None:
    """
    Log a complete RAG query run to MLflow.

    One call = one MLflow Run containing:
      ① All pipeline configuration as params
      ② Performance metrics (latency, source count, text lengths)
      ③ RAG quality metrics — DeepEval (neural) when available, otherwise
         lexical approximations as a fallback
      ④ Full Q&A payload saved as a JSON artifact

    Args:
        query           : the user's question
        answer          : the LLM's response
        sources         : [{"source": "file.pdf", "page": "5"}, ...]
        response_time   : total pipeline duration in seconds
        user_id         : authenticated user who made the request
        deepeval_scores : optional dict from deepeval_service.compute_rag_metrics()
                          keys: answer_relevancy, faithfulness, contextual_relevancy
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_param("chunk_size",          CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",       CHUNK_OVERLAP)
        mlflow.log_param("chunking_strategy",   CHUNKING_STRATEGY)

        mlflow.log_param("embedding_model",     EMBEDDING_MODEL)
        mlflow.log_param("embedding_dim",       EMBEDDING_DIM)
        mlflow.log_param("embedding_normalize", EMBEDDING_NORMALIZE)

        mlflow.log_param("similarity_metric",   SIMILARITY_METRIC)
        mlflow.log_param("k_dense",             K_DENSE)
        mlflow.log_param("k_sparse",            K_SPARSE)
        mlflow.log_param("k_final",             K_FINAL)
        mlflow.log_param("reranker_model",      RERANKER_MODEL)
        mlflow.log_param("retriever_type",      RETRIEVER_TYPE)

        mlflow.log_param("llm_model",           LLM_MODEL)
        mlflow.log_param("llm_temperature",     LLM_TEMPERATURE)
        mlflow.log_param("llm_max_tokens",      str(LLM_MAX_TOKENS))
        mlflow.log_param("llm_top_p",           LLM_TOP_P)
        mlflow.log_param("llm_top_k",           LLM_TOP_K)
        mlflow.log_param("prompt_template",     PROMPT_TEMPLATE)

        mlflow.log_param("user_id",             user_id)

        mlflow.log_metric("response_time_ms",   round(response_time * 1000))
        mlflow.log_metric("num_sources",        len(sources))
        mlflow.log_metric("query_length",       len(query.split()))
        mlflow.log_metric("answer_length",      len(answer.split()))

        if deepeval_scores:

            mlflow.log_metric("de_answer_relevancy",
                              deepeval_scores.get("answer_relevancy", -1.0))
            mlflow.log_metric("de_faithfulness",
                              deepeval_scores.get("faithfulness", -1.0))
            mlflow.log_metric("de_contextual_relevancy",
                              deepeval_scores.get("contextual_relevancy", -1.0))
            mlflow.log_param("quality_metric_source", "deepeval")
        else:

            mlflow.log_metric("answer_relevance", _answer_relevance(query, answer))
            mlflow.log_metric("faithfulness",     _faithfulness(answer, sources))
            mlflow.log_param("quality_metric_source", "lexical_fallback")

        mlflow.log_metric("precision_at_k",     _precision_at_k(sources))
        mlflow.log_metric("recall_at_k",        _recall_at_k(sources))

        payload = {
            "query":         query,
            "answer":        answer,
            "sources":       sources,
            "response_time": round(response_time, 3),
            "user_id":       user_id,
            "pipeline_config": {
                "llm_model":          LLM_MODEL,
                "llm_temperature":    LLM_TEMPERATURE,
                "embedding_model":    EMBEDDING_MODEL,
                "retriever_type":     RETRIEVER_TYPE,
                "k_dense":            K_DENSE,
                "k_sparse":           K_SPARSE,
                "k_final":            K_FINAL,
                "chunk_size":         CHUNK_SIZE,
                "chunk_overlap":      CHUNK_OVERLAP,
                "chunking_strategy":  CHUNKING_STRATEGY,
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name

        mlflow.log_artifact(tmp_path, artifact_path="rag_results")
        os.unlink(tmp_path)                                              
