"""
RAG CHAIN MODULE - Orchestrating the Complete RAG Pipeline
========================================================

LEARNING OBJECTIVES:
1. Understand how all RAG components work together
2. Learn effective prompt engineering for technical domains
3. See how to prevent hallucinations and ensure grounded responses
4. Understand the complete query → context → response flow

KEY CONCEPTS:
- RAG Chain = orchestrates retrieval + generation
- Prompt Engineering = critical for quality responses
- Grounding = ensuring responses are based only on retrieved context
- Source Attribution = providing citations for trustworthiness
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# For local LLM (Ollama)
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# For OpenAI (alternative)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

from .vectorstore import VectorStore
from .retrieval import SmartRetriever

# LLMOps monitoring (graceful import)
try:
    from ..monitoring import RAGMLflowTracker, RAGMetrics, RAGEvaluator
    MONITORING_AVAILABLE = True
except Exception:
    MONITORING_AVAILABLE = False

load_dotenv()

@dataclass
class RAGResponse:
    """
    Structured response from the RAG system.
    """
    answer: str
    sources: List[str]
    confidence_score: float
    retrieved_chunks: int
    query_timestamp: str
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence_score,
            "retrieved_chunks": self.retrieved_chunks,
            "timestamp": self.query_timestamp,
            "processing_time_ms": self.processing_time_ms
        }

class BiomedicRAGChain:
    """
    Complete RAG chain for biomedical equipment support.
    
    WHY CUSTOM CHAIN?
    - Full control over prompt engineering
    - Domain-specific optimizations
    - Better error handling and logging
    - Structured responses with source attribution
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retriever = SmartRetriever(vector_store)
        
        # Initialize LLM based on availability and configuration
        self.llm = self._initialize_llm()
        
        # Create optimized prompt template
        self.prompt_template = self._create_prompt_template()

        # ── LLMOps monitoring ──────────────────────────────────────
        if MONITORING_AVAILABLE:
            try:
                self.mlflow_tracker = RAGMLflowTracker()
                self.metrics = RAGMetrics()
                self.evaluator = RAGEvaluator()
                self._query_counter: int = 0  # step counter for MLflow
                print("📊 LLMOps monitoring initialised (MLflow + Prometheus + DeepEval)")
            except Exception as e:
                print(f"⚠️  LLMOps monitoring disabled: {e}")
                self.mlflow_tracker = None
                self.metrics = None
                self.evaluator = None
        else:
            self.mlflow_tracker = None
            self.metrics = None
            self.evaluator = None
        
        print(f"🔗 RAG Chain initialized")
        print(f"   LLM: {type(self.llm).__name__}")
        print(f"   Retrieval strategy: hybrid")
        print(f"   Vector store: {vector_store.collection_name}")
    
    def _initialize_llm(self):
        """Initialize the best available LLM."""
        llm_model = os.getenv("LLM_MODEL", "llama3.2")
        
        # Try Ollama first (local, privacy-friendly)
        if OLLAMA_AVAILABLE:
            try:
                print(f"🔄 Initializing Ollama with model: {llm_model}")
                llm = Ollama(
                    model=llm_model,
                    temperature=0.1,  # Low temperature for factual responses
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                # Test the connection
                test_response = llm.invoke("Test")
                print("✅ Ollama LLM initialized successfully")
                return llm
            except Exception as e:
                print(f"⚠️  Ollama not available: {e}")
        
        # Fallback to OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            print("🔄 Initializing OpenAI LLM...")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # No LLM available - return mock for testing
        print("⚠️  No LLM available. Using mock responses for testing.")
        return MockLLM()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create optimized prompt template for biomedical technical support.
        
        PROMPT ENGINEERING PRINCIPLES:
        1. Clear role definition (technical assistant)
        2. Strict grounding instructions (no hallucinations)
        3. Structured response format
        4. Source attribution requirements
        5. Domain-specific context
        """
        
        system_prompt = """You are a specialized technical assistant for biomedical laboratory equipment.

CRITICAL INSTRUCTIONS:
1. ONLY answer based on the provided documentation context
2. If information is not in the context, explicitly state: "This information is not available in the provided documentation"
3. Always cite sources using the document references provided
4. Provide step-by-step instructions when applicable
5. Focus on safety and accuracy - incorrect advice could affect patient care
6. Use clear, professional language suitable for laboratory technicians

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed steps if it's a procedural question
- Include relevant safety warnings
- End with source citations in format: [Source: Document X - filename]

CONTEXT DOCUMENTATION:
{context}

USER QUESTION: {question}

RESPONSE:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
    
    def query(self, question: str, retrieval_strategy: str = "hybrid") -> RAGResponse:
        """
        Process a user query through the complete RAG pipeline.
        
        PIPELINE STEPS:
        1. Retrieve relevant documents
        2. Format context for LLM
        3. Generate response using prompt template
        4. Extract sources and calculate confidence
        5. Return structured response
        """
        start_time = datetime.now()
        
        print(f"🔗 Processing RAG query: '{question[:100]}...'")
        
        # Step 1: Retrieve relevant documents
        print("📄 Step 1: Document retrieval...")
        retrieved_docs = self.retriever.retrieve(question, strategy=retrieval_strategy)
        
        if not retrieved_docs:
            return RAGResponse(
                answer="I don't have relevant documentation to answer this question. Please ensure the relevant manuals have been uploaded to the system.",
                sources=[],
                confidence_score=0.0,
                retrieved_chunks=0,
                query_timestamp=datetime.now().isoformat(),
                processing_time_ms=self._calculate_processing_time(start_time)
            )
        
        # Step 2: Format context
        print("📝 Step 2: Context formatting...")
        context = self.retriever.get_context_for_llm(retrieved_docs)
        
        # Step 3: Generate response
        print("🤖 Step 3: Response generation...")
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            response = self.llm.invoke(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
                
        except Exception as e:
            print(f"❌ LLM invocation failed: {e}")
            answer = f"I apologize, but I encountered a technical error while processing your question. Error: {str(e)}"
        
        # Step 4: Extract sources and calculate confidence
        sources = [doc.source_info for doc in retrieved_docs]
        confidence = self._calculate_confidence(retrieved_docs, answer)
        
        processing_time = self._calculate_processing_time(start_time)
        
        response_obj = RAGResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence,
            retrieved_chunks=len(retrieved_docs),
            query_timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
        print(f"✅ RAG processing complete ({processing_time:.0f}ms)")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Sources: {len(sources)}")

        # ── Step 5: LLMOps — log to MLflow + Prometheus ───────────
        self._log_to_monitoring(
            question=question,
            context=context,
            answer=answer,
            sources=sources,
            retrieved_docs=retrieved_docs,
            retrieval_strategy=retrieval_strategy,
            confidence=confidence,
            processing_time=processing_time,
        )
        
        return response_obj

    def _log_to_monitoring(
        self,
        question: str,
        context: str,
        answer: str,
        sources: List[str],
        retrieved_docs: list,
        retrieval_strategy: str,
        confidence: float,
        processing_time: float,
        query_id: Optional[int] = None,
    ) -> None:
        """
        Centralised LLMOps logging — called after every successful query.
        Logs to MLflow (params, metrics, artifacts) and Prometheus (gauges/counters).
        """
        if not (self.mlflow_tracker or self.metrics):
            return

        try:
            # ── DeepEval scores ──────────────────────────────────
            deepeval_scores: Dict[str, float] = {}
            if self.evaluator:
                context_chunks = [doc.content for doc in retrieved_docs]
                deepeval_scores = self.evaluator.evaluate(
                    question=question,
                    answer=answer,
                    context=context_chunks,
                    retrieved_docs=context_chunks,
                )
                print(f"   📐 DeepEval — faithfulness={deepeval_scores.get('faithfulness', 0):.2f}"
                      f"  relevance={deepeval_scores.get('answer_relevance', 0):.2f}"
                      f"  P@k={deepeval_scores.get('precision_k', 0):.2f}"
                      f"  R@k={deepeval_scores.get('recall_k', 0):.2f}")

            # ── Prometheus ───────────────────────────────────────
            if self.metrics:
                self.metrics.record_query(
                    latency_ms=processing_time,
                    confidence=confidence,
                    strategy=retrieval_strategy,
                    success=True,
                    deepeval_scores=deepeval_scores,
                )

            # ── MLflow ───────────────────────────────────────────
            if self.mlflow_tracker:
                self._query_counter += 1
                config = RAGMLflowTracker.build_config_from_env()
                # Inject prompt length into config
                config["llm"]["prompt_length"] = len(str(self.prompt_template))

                with self.mlflow_tracker.start_rag_run(
                    config=config if self._query_counter == 1 else None,
                    run_name=f"query-{self._query_counter}",
                ) as run:
                    # Log context + response as artifact
                    self.mlflow_tracker.log_query(
                        run=run,
                        question=question,
                        context=context,
                        answer=answer,
                        sources=sources,
                        query_id=query_id or self._query_counter,
                    )

                    # Log numeric metrics
                    mlflow_metrics = {
                        "confidence": confidence,
                        "latency_ms": processing_time,
                        "retrieved_chunks": float(len(retrieved_docs)),
                        **{k: float(v) for k, v in deepeval_scores.items()},
                    }
                    self.mlflow_tracker.log_metrics(
                        run=run,
                        metrics=mlflow_metrics,
                        step=self._query_counter,
                    )

        except Exception as e:
            # Never let monitoring errors crash a query
            print(f"   ⚠️  LLMOps logging error (non-fatal): {e}")
    
    def _calculate_confidence(self, retrieved_docs: List, answer: str) -> float:
        """
        Calculate confidence score based on retrieval quality and response.
        
        CONFIDENCE FACTORS:
        - Average similarity score of retrieved documents
        - Number of documents retrieved
        - Presence of "I don't know" phrases in response
        """
        if not retrieved_docs:
            return 0.0
        
        # Base confidence from retrieval scores
        avg_similarity = sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
        
        # Penalty for uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "not sure", "cannot determine", 
            "not available in the documentation", "no information"
        ]
        
        answer_lower = answer.lower()
        uncertainty_penalty = 0.0
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                uncertainty_penalty += 0.2
        
        # Final confidence (between 0 and 1)
        confidence = max(0.0, avg_similarity - uncertainty_penalty)
        
        return min(1.0, confidence)
    
    def _calculate_processing_time(self, start_time: datetime) -> float:
        """Calculate processing time in milliseconds."""
        return (datetime.now() - start_time).total_seconds() * 1000
    
    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Process multiple queries efficiently."""
        print(f"🔗 Processing {len(questions)} queries in batch...")
        
        responses = []
        for i, question in enumerate(questions, 1):
            print(f"\nBatch query {i}/{len(questions)}")
            response = self.query(question)
            responses.append(response)
        
        print(f"✅ Batch processing complete: {len(responses)} responses")
        return responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status information about the RAG system."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "llm_model": type(self.llm).__name__,
            "retrieval_config": {
                "top_k": self.retriever.top_k,
                "final_k": self.retriever.final_k,
                "similarity_threshold": self.retriever.similarity_threshold
            },
            "prompt_template_length": len(str(self.prompt_template))
        }

class MockLLM:
    """Mock LLM for testing when no real LLM is available."""
    
    def invoke(self, prompt: str) -> str:
        return """Based on the provided documentation, I can see information about equipment maintenance procedures. 

For laboratory equipment calibration:
1. Follow the manufacturer's specifications in the equipment manual
2. Perform calibration checks according to the scheduled maintenance intervals
3. Document all calibration results for quality assurance

Please note: This is a mock response for testing purposes. A real LLM would provide more specific guidance based on the actual retrieved documentation.

[Source: Mock Response - No actual documents processed]"""

# Factory functions for easy setup
def create_rag_chain(vector_store: VectorStore) -> BiomedicRAGChain:
    """Create a complete RAG chain."""
    return BiomedicRAGChain(vector_store)

def test_rag_chain():
    """Test the complete RAG chain with sample data."""
    print("🧪 TESTING COMPLETE RAG CHAIN:")
    print("=" * 60)
    
    from .vectorstore import VectorStore
    from langchain.schema import Document
    
    # Create test data
    store = VectorStore(collection_name="test_rag_chain")
    
    sample_docs = [
        Document(
            page_content="Centrifuge calibration procedure: 1) Turn off power 2) Remove rotor 3) Clean chamber 4) Reinstall rotor 5) Run calibration cycle 6) Verify RPM accuracy using external tachometer 7) Document results in maintenance log",
            metadata={"filename": "centrifuge_manual.pdf", "section": "calibration", "likely_pages": [25]}
        ),
        Document(
            page_content="Spectrophotometer daily startup: Check wavelength accuracy at 486nm and 656nm. If deviation >2nm, perform full calibration. Replace deuterium lamp if intensity <80% of specification.",
            metadata={"filename": "spectro_guide.pdf", "section": "daily_procedures", "likely_pages": [12]}
        ),
        Document(
            page_content="Safety notice: Never attempt calibration while equipment is powered on. Always use appropriate PPE including safety glasses and lab coat. Ensure proper ventilation in work area.",
            metadata={"filename": "safety_manual.pdf", "section": "calibration_safety", "likely_pages": [8]}
        )
    ]
    
    store.add_documents(sample_docs)
    
    # Create and test RAG chain
    rag_chain = BiomedicRAGChain(store)
    
    # Test questions
    test_questions = [
        "How do I calibrate a centrifuge?",
        "What safety precautions should I take during equipment calibration?",
        "How often should I replace the deuterium lamp in a spectrophotometer?",
        "What is the procedure for spacecraft maintenance?"  # Should return "no information"
    ]
    
    print("\nTesting RAG responses:")
    print("=" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        response = rag_chain.query(question)
        
        print(f"✅ Answer: {response.answer[:200]}...")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print(f"📄 Sources: {len(response.sources)}")
        print(f"⏱️  Processing time: {response.processing_time_ms:.0f}ms")
    
    # Test system status
    status = rag_chain.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Documents in store: {status['vector_store']['total_documents']}")
    print(f"   LLM model: {status['llm_model']}")
    print(f"   Retrieval top_k: {status['retrieval_config']['top_k']}")
    
    # Cleanup
    store.delete_collection()
    print("\n🧹 Test data cleaned up")

if __name__ == "__main__":
    test_rag_chain()