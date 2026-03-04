import asyncio
from typing import Optional

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_community.chat_models import ChatOllama

class OllamaJudge(DeepEvalBaseLLM):
    """
    A DeepEval-compatible LLM judge backed by a local Ollama model.

    How it works:
      DeepEval calls judge.generate(prompt) internally when computing metrics.
      We delegate those calls to ChatOllama (LangChain / Ollama local API).
    """

    def __init__(self, model: str = "mistral"):
        """
        Args:
            model: Name of the Ollama model to use as the judge.
                   "mistral" is the same model used for generation — consistent.
        """
        self.model     = model
        self._client   = None                                   

    def load_model(self):
        """
        Called by DeepEval once before any generate() calls.
        Returns the underlying model object (ChatOllama here).
        """
        if self._client is None:

            self._client = ChatOllama(model=self.model, temperature=0)
        return self._client

    def generate(self, prompt: str) -> str:
        """
        Synchronous text generation — called by DeepEval metric.measure().

        Args:
            prompt: The full evaluation prompt assembled by DeepEval
                    (contains the query, answer, context, and evaluation
                    instructions in a structured format).
        Returns:
            The judge's response as a plain string.
        """
        client = self.load_model()
        response = client.invoke(prompt)

        return response.content

    async def a_generate(self, prompt: str) -> str:
        """
        Async version — required by DeepEvalBaseLLM interface.
        We run the synchronous generate() in a thread pool so we don't block.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self) -> str:
        """Returns a human-readable name for logging purposes."""
        return f"ollama/{self.model}"

_judge = OllamaJudge(model="mistral")

def compute_rag_metrics(
    query:    str,
    answer:   str,
    contexts: list[str],
    threshold: float = 0.5,
) -> dict:
    """
    Run all three DeepEval metrics for one RAG query.

    Args:
        query     : The user's original question (input to the RAG pipeline)
        answer    : The LLM's answer (actual_output)
        contexts  : List of retrieved chunk texts (retrieval_context).
                    These are the doc.page_content strings from rag_engine.py.
        threshold : Scores above this are considered "passing".
                    DeepEval uses this to decide pass/fail per metric.

    Returns:
        {
            "answer_relevancy":      float,   # 0.0 → 1.0
            "faithfulness":          float,   # 0.0 → 1.0
            "contextual_relevancy":  float,   # 0.0 → 1.0
        }

    Note:
        If any metric fails (model unavailable, parsing error), its score
        defaults to -1.0.  This is intentional: -1.0 clearly signals
        "metric not computed" and is easy to filter in MLflow queries.
    """

    test_case = LLMTestCase(
        input             = query,                                
        actual_output     = answer,                                 
        retrieval_context = contexts,                                               
    )

    scores = {
        "answer_relevancy":     -1.0,
        "faithfulness":         -1.0,
        "contextual_relevancy": -1.0,
    }

    try:
        m = AnswerRelevancyMetric(
            threshold = threshold,
            model     = _judge,
            verbose_mode = False,
        )
        m.measure(test_case)
        scores["answer_relevancy"] = round(m.score, 3)
    except Exception as e:
        print(f"[DeepEval] AnswerRelevancyMetric failed: {e}")

    try:
        m = FaithfulnessMetric(
            threshold = threshold,
            model     = _judge,
            verbose_mode = False,
        )
        m.measure(test_case)
        scores["faithfulness"] = round(m.score, 3)
    except Exception as e:
        print(f"[DeepEval] FaithfulnessMetric failed: {e}")

    try:
        m = ContextualRelevancyMetric(
            threshold = threshold,
            model     = _judge,
            verbose_mode = False,
        )
        m.measure(test_case)
        scores["contextual_relevancy"] = round(m.score, 3)
    except Exception as e:
        print(f"[DeepEval] ContextualRelevancyMetric failed: {e}")

    return scores
