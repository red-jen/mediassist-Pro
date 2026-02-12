"""
QUERY ROUTES - RAG Query Processing API
======================================

LEARNING OBJECTIVES:
1. Understand how to expose RAG functionality via REST API
2. Learn asynchronous processing for better performance
3. See query logging and analytics implementation
4. Understand structured response formatting

KEY CONCEPTS:
- RAG endpoint = user query → retrieval → generation → response
- Async processing = non-blocking I/O for better performance
- Query logging = track usage and improve system
- Response streaming = handle long-running LLM responses
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ...db import get_database_session, User, Query as QueryModel
from ...core import get_current_user, AuthorizationService, RAGSystemError
from ...rag import create_rag_chain, create_vector_store, RAGResponse

# Create router
router = APIRouter(prefix="/query", tags=["Query Processing"])

# === REQUEST/RESPONSE MODELS ===

class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(
        ..., 
        min_length=10, 
        max_length=1000,
        description="User question about biomedical equipment"
    )
    retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: naive, expanded, or hybrid"
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response"
    )
    max_response_length: Optional[int] = Field(
        default=None,
        description="Maximum response length in characters"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do I calibrate a centrifuge according to safety protocols?",
                "retrieval_strategy": "hybrid",
                "include_sources": True
            }
        }
    }

class QueryResponse(BaseModel):
    """Query response model."""
    query_id: int = Field(..., description="Unique query identifier")
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Answer confidence (0.0-1.0)")
    sources: List[str] = Field(..., description="Source documents used")
    retrieved_chunks: int = Field(..., description="Number of retrieved document chunks")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: str = Field(..., description="Query timestamp")
    retrieval_strategy: str = Field(..., description="Retrieval strategy used")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": 123,
                "question": "How do I calibrate a centrifuge?",
                "answer": "To calibrate a centrifuge, follow these steps: 1) Turn off power...",
                "confidence_score": 0.85,
                "sources": ["centrifuge_manual.pdf (pages 15-16)", "safety_guide.pdf (page 8)"],
                "retrieved_chunks": 3,
                "processing_time_ms": 1250.5,
                "timestamp": "2024-02-09T10:30:00Z",
                "retrieval_strategy": "hybrid"
            }
        }
    }

class QueryHistory(BaseModel):
    """Query history item."""
    id: int
    question: str
    answer: str
    confidence_score: float
    sources_count: int
    created_at: datetime
    processing_time_ms: float
    user_rating: Optional[int] = None
    
    class Config:
        from_attributes = True

class QueryFeedback(BaseModel):
    """Query feedback model."""
    query_id: int = Field(..., description="Query ID to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback: Optional[str] = Field(default=None, max_length=500, description="Optional feedback text")

class QueryStats(BaseModel):
    """Query statistics model."""
    total_queries: int
    avg_confidence: float
    avg_processing_time_ms: float
    top_confidence_queries: int
    recent_queries: int
    user_queries: int

# === CORE RAG FUNCTIONALITY ===

# Initialize RAG system (this would be done at startup)
vector_store = None
rag_chain = None

async def get_rag_system():
    """Get or initialize RAG system."""
    global vector_store, rag_chain
    
    if vector_store is None:
        vector_store = create_vector_store()
        
    if rag_chain is None:
        rag_chain = create_rag_chain(vector_store)
    
    return rag_chain

async def log_query_to_database(
    db: Session,
    user: User,
    query_request: QueryRequest,
    rag_response: RAGResponse,
    background_tasks: BackgroundTasks
):
    """Log query to database asynchronously."""
    
    def save_query():
        query_record = QueryModel(
            query_text=query_request.question,
            response_text=rag_response.answer,
            processing_time_ms=rag_response.processing_time_ms,
            confidence_score=rag_response.confidence_score,
            retrieved_chunks=rag_response.retrieved_chunks,
            retrieval_strategy=query_request.retrieval_strategy,
            sources_used=rag_response.sources,
            user_id=user.id
        )
        
        db.add(query_record)
        db.commit()
        db.refresh(query_record)
        return query_record
    
    # Execute in background
    background_tasks.add_task(save_query)

# === QUERY ENDPOINTS ===

@router.post("/ask", response_model=QueryResponse, summary="Ask RAG System")
async def ask_question(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """
    Process user question through RAG system.
    
    RAG PROCESSING FLOW:
    1. Validate user permissions
    2. Initialize RAG system components
    3. Process query through retrieval → generation
    4. Log query and response
    5. Return structured response
    
    **Requires**: Valid authentication
    **Returns**: Generated answer with sources and metadata
    """
    
    # Check user permissions
    if not AuthorizationService.has_permission(current_user, "search_documents"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to query the system"
        )
    
    try:
        # Validate retrieval strategy
        valid_strategies = ["naive", "expanded", "hybrid"]
        if query_request.retrieval_strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid retrieval strategy. Must be one of: {valid_strategies}"
            )
        
        # Get RAG system
        rag_system = await get_rag_system()
        
        # Check if vector store has documents
        store_stats = rag_system.vector_store.get_collection_stats()
        if store_stats["total_documents"] == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents found in the knowledge base. Please upload documents first."
            )
        
        # Process query
        print(f"🔍 Processing query from user {current_user.username}: {query_request.question}")
        
        rag_response = rag_system.query(
            question=query_request.question,
            retrieval_strategy=query_request.retrieval_strategy
        )
        
        # Truncate response if requested
        answer = rag_response.answer
        if query_request.max_response_length and len(answer) > query_request.max_response_length:
            answer = answer[:query_request.max_response_length] + "..."
        
        # Prepare sources (filter if requested)
        sources = rag_response.sources if query_request.include_sources else []
        
        # Save to database (background task)
        query_record = QueryModel(
            query_text=query_request.question,
            response_text=answer,
            processing_time_ms=rag_response.processing_time_ms,
            confidence_score=rag_response.confidence_score,
            retrieved_chunks=rag_response.retrieved_chunks,
            retrieval_strategy=query_request.retrieval_strategy,
            sources_used=sources,
            user_id=current_user.id
        )
        
        db.add(query_record)
        db.commit()
        db.refresh(query_record)
        
        # Return structured response
        return QueryResponse(
            query_id=query_record.id,
            question=query_request.question,
            answer=answer,
            confidence_score=rag_response.confidence_score,
            sources=sources,
            retrieved_chunks=rag_response.retrieved_chunks,
            processing_time_ms=rag_response.processing_time_ms,
            timestamp=rag_response.query_timestamp,
            retrieval_strategy=query_request.retrieval_strategy
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ RAG query failed: {str(e)}")
        raise RAGSystemError(
            message=f"Failed to process query: {str(e)}",
            component="query_processing"
        )

@router.get("/history", response_model=List[QueryHistory], summary="Get Query History")
async def get_query_history(
    limit: int = 20,
    skip: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """
    Get user's query history.
    
    **Requires**: Valid authentication
    **Returns**: List of previous queries and responses
    """
    
    # Users can see their own queries, admins can see all
    if current_user.is_admin:
        queries = db.query(QueryModel).offset(skip).limit(limit).all()
    else:
        queries = db.query(QueryModel).filter(
            QueryModel.user_id == current_user.id
        ).offset(skip).limit(limit).all()
    
    # Convert to response format
    history_items = []
    for query in queries:
        history_items.append(QueryHistory(
            id=query.id,
            question=query.query_text,
            answer=query.response_text[:200] + "..." if len(query.response_text) > 200 else query.response_text,
            confidence_score=query.confidence_score or 0.0,
            sources_count=len(query.sources_used) if query.sources_used else 0,
            created_at=query.created_at,
            processing_time_ms=query.processing_time_ms or 0.0,
            user_rating=query.user_rating
        ))
    
    return history_items

@router.post("/feedback", summary="Provide Query Feedback")
async def provide_feedback(
    feedback: QueryFeedback,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """
    Provide feedback on a query response.
    
    **Purpose**: Collect user feedback to improve RAG system
    **Requires**: Valid authentication
    **Returns**: Feedback confirmation
    """
    
    # Check if current user has permission to rate
    if not AuthorizationService.has_permission(current_user, "rate_responses"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to rate responses"
        )
    
    # Find the query
    query = db.query(QueryModel).filter(QueryModel.id == feedback.query_id).first()
    
    if not query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    # Check if user owns the query or is admin
    if query.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only rate your own queries"
        )
    
    # Update feedback
    query.user_rating = feedback.rating
    query.user_feedback = feedback.feedback
    
    db.commit()
    
    return {
        "message": "Feedback recorded successfully",
        "query_id": feedback.query_id,
        "rating": feedback.rating,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/stats", response_model=QueryStats, summary="Get Query Statistics")
async def get_query_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """
    Get query statistics.
    
    **Requires**: Valid authentication
    **Returns**: Query usage and performance statistics
    """
    
    # Determine query scope (user's queries vs all queries)
    if current_user.is_admin:
        base_query = db.query(QueryModel)
        scope = "system"
    else:
        base_query = db.query(QueryModel).filter(QueryModel.user_id == current_user.id)
        scope = "user"
    
    # Calculate statistics
    from sqlalchemy import func
    
    total_queries = base_query.count()
    
    if total_queries == 0:
        return QueryStats(
            total_queries=0,
            avg_confidence=0.0,
            avg_processing_time_ms=0.0,
            top_confidence_queries=0,
            recent_queries=0,
            user_queries=0 if scope == "system" else 0
        )
    
    # Average confidence
    avg_confidence = db.query(func.avg(QueryModel.confidence_score)).filter(
        base_query.whereclause
    ).scalar() or 0.0
    
    # Average processing time
    avg_processing_time = db.query(func.avg(QueryModel.processing_time_ms)).filter(
        base_query.whereclause
    ).scalar() or 0.0
    
    # High confidence queries (>0.8)
    high_confidence_queries = base_query.filter(QueryModel.confidence_score > 0.8).count()
    
    # Recent queries (last 24 hours)
    from datetime import timedelta
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_queries = base_query.filter(QueryModel.created_at > recent_cutoff).count()
    
    # User's own queries (for admin view)
    user_queries = 0
    if current_user.is_admin:
        user_queries = db.query(QueryModel).filter(QueryModel.user_id == current_user.id).count()
    
    return QueryStats(
        total_queries=total_queries,
        avg_confidence=round(avg_confidence, 3),
        avg_processing_time_ms=round(avg_processing_time, 1),
        top_confidence_queries=high_confidence_queries,
        recent_queries=recent_queries,
        user_queries=user_queries if scope == "system" else total_queries
    )

@router.get("/system-status", summary="Get RAG System Status")
async def get_system_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get RAG system status and health.
    
    **Requires**: Valid authentication
    **Returns**: System components status and configuration
    """
    
    try:
        # Get RAG system
        rag_system = await get_rag_system()
        
        # Get system status
        system_status = rag_system.get_system_status()
        
        # Add user context
        system_status["user_info"] = {
            "username": current_user.username,
            "role": current_user.role,
            "permissions": AuthorizationService.ROLE_PERMISSIONS.get(current_user.role, [])
        }
        
        system_status["timestamp"] = datetime.utcnow().isoformat()
        
        return system_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

if __name__ == "__main__":
    # Test query models
    print("🧪 TESTING QUERY ROUTE MODELS:")
    print("=" * 40)
    
    # Test query request
    query_req = QueryRequest(
        question="How do I calibrate laboratory equipment?",
        retrieval_strategy="hybrid"
    )
    print(f"Query Request: {query_req.model_dump()}")
    
    # Test feedback model
    feedback = QueryFeedback(query_id=1, rating=5, feedback="Very helpful response!")
    print(f"Feedback: {feedback.model_dump()}")
    
    print("\n✅ Query route models test complete")