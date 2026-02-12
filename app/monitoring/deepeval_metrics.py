"""
DEEPEVAL RAG METRICS
====================

Computes the four RAG quality metrics required by the project brief:

  1. Answer Relevance  – is the answer actually relevant to the question?
  2. Faithfulness      – is every claim in the answer grounded in the context?
  3. Precision@k       – what fraction of retrieved chunks are truly relevant?
  4. Recall@k          – what fraction of all relevant chunks were retrieved?

DeepEval documentation: https://docs.confident-ai.com/

Usage:
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(
        question="How do I calibrate a centrifuge?",
        answer="Turn off power, remove rotor...",
        context=["Doc1 text...", "Doc2 text..."],
        retrieved_docs=["Doc1 text...", "Doc2 text..."],
        relevant_docs=["Doc1 text..."],   # ground truth (optional)
    )
    # scores = {"faithfulness": 0.92, "answer_relevance": 0.88, ...}
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class RAGEvaluator:
    """
    Wrapper around DeepEval for computing RAG-specific quality metrics.

    GRACEFUL DEGRADATION: If DeepEval or its dependencies are missing
    (e.g. no OpenAI key in some environments), falls back to lightweight
    heuristic scores so the rest of the pipeline keeps working.
    """

    def __init__(self):
        self._deepeval_available = self._check_deepeval()
        if self._deepeval_available:
            print("✅ DeepEval available — full RAG evaluation enabled")
        else:
            print("⚠️  DeepEval not available — using heuristic evaluation fallback")

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        retrieved_docs: Optional[List[str]] = None,
        relevant_docs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all four RAG metrics and return as a float dict.

        Args:
            question      : The user's original question
            answer        : LLM-generated answer
            context       : List of context strings fed to the LLM
            retrieved_docs: Chunks returned by the retriever  (for P@k / R@k)
            relevant_docs : Ground-truth relevant chunks        (for P@k / R@k)

        Returns:
            {
              "faithfulness":     0.0–1.0,
              "answer_relevance": 0.0–1.0,
              "precision_k":      0.0–1.0,
              "recall_k":         0.0–1.0,
            }
        """
        if self._deepeval_available:
            return self._evaluate_with_deepeval(
                question, answer, context, retrieved_docs, relevant_docs
            )
        return self._heuristic_evaluate(
            question, answer, context, retrieved_docs, relevant_docs
        )

    # ─────────────────────────────────────────────────────────────
    # DeepEval implementation
    # ─────────────────────────────────────────────────────────────
    def _evaluate_with_deepeval(
        self,
        question: str,
        answer: str,
        context: List[str],
        retrieved_docs: Optional[List[str]],
        relevant_docs: Optional[List[str]],
    ) -> Dict[str, float]:
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
            )
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=context,
                expected_output=relevant_docs[0] if relevant_docs else answer,
            )

            scores: Dict[str, float] = {}

            # 1. Answer Relevance
            try:
                relevance_metric = AnswerRelevancyMetric(threshold=0.5, verbose_mode=False)
                relevance_metric.measure(test_case)
                scores["answer_relevance"] = float(relevance_metric.score or 0.0)
            except Exception as e:
                scores["answer_relevance"] = self._heuristic_relevance(question, answer)
                print(f"   ⚠️  DeepEval AnswerRelevancy failed ({e}), using heuristic")

            # 2. Faithfulness
            try:
                faithfulness_metric = FaithfulnessMetric(threshold=0.5, verbose_mode=False)
                faithfulness_metric.measure(test_case)
                scores["faithfulness"] = float(faithfulness_metric.score or 0.0)
            except Exception as e:
                scores["faithfulness"] = self._heuristic_faithfulness(answer, context)
                print(f"   ⚠️  DeepEval Faithfulness failed ({e}), using heuristic")

            # 3. Precision@k
            if retrieved_docs and relevant_docs:
                try:
                    precision_metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=False)
                    precision_metric.measure(test_case)
                    scores["precision_k"] = float(precision_metric.score or 0.0)
                except Exception as e:
                    scores["precision_k"] = self._heuristic_precision(retrieved_docs, relevant_docs)
                    print(f"   ⚠️  DeepEval Precision@k failed ({e}), using heuristic")

                try:
                    recall_metric = ContextualRecallMetric(threshold=0.5, verbose_mode=False)
                    recall_metric.measure(test_case)
                    scores["recall_k"] = float(recall_metric.score or 0.0)
                except Exception as e:
                    scores["recall_k"] = self._heuristic_recall(retrieved_docs, relevant_docs)
                    print(f"   ⚠️  DeepEval Recall@k failed ({e}), using heuristic")
            else:
                scores["precision_k"] = self._heuristic_precision(retrieved_docs or [], relevant_docs or [])
                scores["recall_k"] = self._heuristic_recall(retrieved_docs or [], relevant_docs or [])

            return scores

        except Exception as e:
            print(f"   ❌ DeepEval evaluation failed: {e}")
            return self._heuristic_evaluate(question, answer, context, retrieved_docs, relevant_docs)

    # ─────────────────────────────────────────────────────────────
    # Heuristic fallback (no external LLM required)
    # ─────────────────────────────────────────────────────────────
    def _heuristic_evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        retrieved_docs: Optional[List[str]],
        relevant_docs: Optional[List[str]],
    ) -> Dict[str, float]:
        """Lightweight word-overlap heuristics — no API calls needed."""
        return {
            "answer_relevance": self._heuristic_relevance(question, answer),
            "faithfulness": self._heuristic_faithfulness(answer, context),
            "precision_k": self._heuristic_precision(retrieved_docs or [], relevant_docs or []),
            "recall_k": self._heuristic_recall(retrieved_docs or [], relevant_docs or []),
        }

    @staticmethod
    def _heuristic_relevance(question: str, answer: str) -> float:
        """Word-overlap between question and answer (rough proxy)."""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = q_words & a_words
        if not q_words:
            return 0.0
        # Penalize "not available" answers
        if any(p in answer.lower() for p in ["not available", "i don't know", "cannot"]):
            return max(0.1, len(overlap) / len(q_words) * 0.5)
        return min(1.0, len(overlap) / max(1, len(q_words)) * 1.5)

    @staticmethod
    def _heuristic_faithfulness(answer: str, context: List[str]) -> float:
        """Fraction of answer words present in provided context."""
        if not context:
            return 0.0
        combined_context = " ".join(context).lower()
        answer_words = [w for w in answer.lower().split() if len(w) > 4]
        if not answer_words:
            return 0.5
        found = sum(1 for w in answer_words if w in combined_context)
        return round(found / len(answer_words), 4)

    @staticmethod
    def _heuristic_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Precision@k: |retrieved ∩ relevant| / |retrieved|"""
        if not retrieved:
            return 0.0
        if not relevant:
            return 0.5  # unknown ground truth → neutral
        relevant_set = {r[:50] for r in relevant}
        hits = sum(1 for d in retrieved if d[:50] in relevant_set)
        return round(hits / len(retrieved), 4)

    @staticmethod
    def _heuristic_recall(retrieved: List[str], relevant: List[str]) -> float:
        """Recall@k: |retrieved ∩ relevant| / |relevant|"""
        if not relevant:
            return 0.5  # unknown ground truth → neutral
        retrieved_set = {d[:50] for d in retrieved}
        hits = sum(1 for r in relevant if r[:50] in retrieved_set)
        return round(hits / len(relevant), 4)

    @staticmethod
    def _check_deepeval() -> bool:
        try:
            import deepeval  # noqa: F401
            return True
        except ImportError:
            return False
