# DeepEval — RAG Quality Evaluation

## What Is It?

DeepEval is a **test framework for LLM outputs**.  
Its job: answer *"was this RAG response any good?"* using another LLM as the judge (LLM-as-judge pattern).

**The problem with word-count metrics:**
```python
# Old lexical approach — scores based on word overlap
query_words  = {"what", "is", "elisa", "used", "for"}
answer_words = {"elisa", "is", "not", "my", "concern"}
score = len(query_words & answer_words) / len(query_words)
# → 0.4  ✓ scored OK but the answer is useless — doesn't understand meaning
```

DeepEval fixes this by having **Mistral read and semantically judge** the answer.

---

## How It Works

```
Input to DeepEval:
  - input              = user's original question
  - actual_output      = the LLM's answer
  - retrieval_context  = the actual text of the top-k retrieved chunks

DeepEval builds an LLMTestCase, runs each metric, returns a score 0.0 → 1.0
```

---

## 3 Metrics We Use

### 1. `AnswerRelevancyMetric` → `de_answer_relevancy`
> "Does the answer actually address the question?"

The judge extracts **statements** from the answer and checks if each one is relevant to the query.  
`score = relevant_statements / total_statements`

```
Query:  "What is the recommended dosage of ibuprofen for adults?"
Answer: "Ibuprofen is an NSAID used to treat pain and inflammation."

Judge: "Describes what it is — does NOT mention dosage."
→ de_answer_relevancy = 0.1   ✗ correctly low

─────────────────────────────────────────────────────────────
Answer: "The recommended adult dosage is 400mg every 4–6 hours,
         not exceeding 1200mg per day."

Judge: "Directly answers the dosage question."
→ de_answer_relevancy = 0.95  ✓ correctly high
```

---

### 2. `FaithfulnessMetric` → `de_faithfulness`
> "Is every claim in the answer supported by the retrieved chunks?"

The judge extracts **factual claims** from the answer, then checks each one against the context.  
`score = grounded_claims / total_claims`

```
Retrieved context: "Ibuprofen should not be taken on an empty stomach."

Answer: "Ibuprofen should be taken with food. It also cures infections."

Claims extracted by judge:
  Claim 1: "take with food"   → ✓ found in context
  Claim 2: "cures infections" → ✗ NOT in context (hallucination!)

→ de_faithfulness = 0.5
```

> This is the **anti-hallucination metric** — the most critical one for a medical app.

---

### 3. `ContextualRelevancyMetric` → `de_contextual_relevancy`
> "Were the right chunks retrieved from the PDF in the first place?"

The judge reads each chunk and decides: *"Is this chunk relevant to the question?"*  
`score = relevant_chunks / total_chunks`

```
Query: "What is the sterilization protocol for ELISA plates?"

Chunk 1: "ELISA plates must be washed 3 times with PBS buffer..."  ✓ relevant
Chunk 2: "Storage temperature for reagents is 4°C..."              ✓ relevant
Chunk 3: "The company was founded in 1987 by Dr. Schmidt..."       ✗ irrelevant

→ de_contextual_relevancy = 0.67
```

---

## What Each Low Score Tells You

| Metric low | Root cause | What to fix |
|---|---|---|
| `de_answer_relevancy` | LLM answered the wrong thing | Improve prompt / query rewriting |
| `de_faithfulness` | LLM is hallucinating | Tighten prompt instructions |
| `de_contextual_relevancy` | Wrong chunks retrieved | Tune k values, embedding model, BM25 weights |

Each metric points to a **different layer** of the RAG pipeline.

---

## Our Implementation

### File: `backend/services/deepeval_service.py`

```python
class OllamaJudge(DeepEvalBaseLLM):
    # Wraps ChatOllama(mistral) so DeepEval uses our local model
    # instead of calling OpenAI — no API key needed
    def generate(self, prompt: str) -> str:
        return ChatOllama(model="mistral", temperature=0).invoke(prompt).content

def compute_rag_metrics(query, answer, contexts) -> dict:
    test_case = LLMTestCase(
        input             = query,
        actual_output     = answer,
        retrieval_context = contexts,   # raw chunk page_content strings
    )
    # Runs all 3 metrics → returns scores as floats
    return {
        "answer_relevancy":     ...,  # 0.0 → 1.0
        "faithfulness":         ...,  # 0.0 → 1.0
        "contextual_relevancy": ...,  # 0.0 → 1.0
    }
```

### Data Flow Per Query

```
rag_engine.ask()
  └── returns: answer + sources + contexts (chunk texts)
        ↓
deepeval_service.compute_rag_metrics(query, answer, contexts)
  ├── OllamaJudge → Mistral judges each metric locally
  ├── AnswerRelevancyMetric    → de_answer_relevancy
  ├── FaithfulnessMetric       → de_faithfulness
  └── ContextualRelevancyMetric → de_contextual_relevancy
        ↓
mlflow_service.log_rag_query(..., deepeval_scores={...})
  └── logged as de_* metrics in MLflow UI (http://localhost:5000)
```

### Why `contexts` (not just `sources`)?

`sources` = `[{"source": "file.pdf", "page": "5"}]` — just metadata, no text.  
`contexts` = `["ELISA plates must be washed 3 times..."]` — the actual chunk content.

FaithfulnessMetric and ContextualRelevancyMetric need the **text** to judge against, not just filenames.

---

## Fallback Behaviour

If DeepEval fails (Ollama down, parsing error):
- Score defaults to `-1.0` — clearly signals "not computed"  
- MLflow logs `quality_metric_source = "lexical_fallback"` instead of `"deepeval"`  
- The API still returns the answer normally — DeepEval is non-blocking

---

## Files Changed

| File | Change |
|---|---|
| `rag_engine.py` | `ask()` now returns `"contexts"` key |
| `backend/services/deepeval_service.py` | New — `OllamaJudge` + `compute_rag_metrics()` |
| `backend/services/rag_service.py` | Calls `compute_rag_metrics()`, passes scores to MLflow |
| `backend/services/mlflow_service.py` | Logs `de_*` metrics, falls back to lexical if unavailable |
