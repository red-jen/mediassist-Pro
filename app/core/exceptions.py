"""
EXCEPTION HANDLING & ERROR MANAGEMENT
====================================

LEARNING OBJECTIVES:
1. Understand centralized exception handling in FastAPI
2. Learn custom exception classes for different error types
3. See structured error responses with proper HTTP status codes
4. Understand error logging and monitoring

KEY CONCEPTS:
- Custom exceptions for domain-specific errors
- Exception handlers for consistent error responses
- Error logging for debugging and monitoring
- User-friendly error messages vs. technical details
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediAssistException(Exception):
    """
    Base exception class for MediAssist Pro application.
    
    WHY CUSTOM EXCEPTIONS?
    - Domain-specific error handling
    - Consistent error structure
    - Better error categorization
    - Easier debugging and monitoring
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class AuthenticationError(MediAssistException):
    """Authentication-related errors."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTH_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class AuthorizationError(MediAssistException):
    """Authorization-related errors."""
    
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHZ_ERROR", 
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )

class ValidationError(MediAssistException):
    """Data validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=validation_details
        )

class DocumentProcessingError(MediAssistException):
    """Document processing and RAG-related errors."""
    
    def __init__(self, message: str, document_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        processing_details = details or {}
        if document_id:
            processing_details["document_id"] = document_id
            
        super().__init__(
            message=message,
            error_code="DOC_PROCESSING_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=processing_details
        )

class RAGSystemError(MediAssistException):
    """RAG system errors (embeddings, retrieval, generation)."""
    
    def __init__(self, message: str, component: str = "unknown", details: Optional[Dict[str, Any]] = None):
        rag_details = details or {}
        rag_details["component"] = component
        
        super().__init__(
            message=message,
            error_code="RAG_SYSTEM_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=rag_details
        )

class DatabaseError(MediAssistException):
    """Database-related errors."""
    
    def __init__(self, message: str, operation: str = "unknown", details: Optional[Dict[str, Any]] = None):
        db_details = details or {}
        db_details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=db_details
        )

class FileUploadError(MediAssistException):
    """File upload and processing errors."""
    
    def __init__(self, message: str, filename: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        upload_details = details or {}
        if filename:
            upload_details["filename"] = filename
            
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=upload_details
        )

class RateLimitError(MediAssistException):
    """Rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )

# Exception Handler Functions
async def mediassist_exception_handler(request: Request, exc: MediAssistException) -> JSONResponse:
    """
    Handle custom MediAssist exceptions.
    
    STRUCTURED ERROR RESPONSE:
    - Consistent JSON structure
    - Error codes for programmatic handling
    - User-friendly messages
    - Technical details for debugging (in non-production)
    """
    
    # Log the error
    logger.error(
        f"MediAssist Exception: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    # Prepare response
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "status_code": exc.status_code,
            "timestamp": str(datetime.now()),
            "path": request.url.path
        }
    }
    
    # Add details if available (be careful with sensitive data)
    if exc.details:
        error_response["error"]["details"] = exc.details
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    SECURITY NOTE:
    - Don't expose internal error details to users
    - Log full stack traces for debugging
    - Return generic error messages
    """
    
    # Get full traceback for logging
    tb = traceback.format_exc()
    
    # Log the error with full details
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": tb
        }
    )
    
    # Return generic error response (don't expose internal details)
    error_response = {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "timestamp": str(datetime.now()),
            "path": request.url.path
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )

async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle FastAPI validation exceptions."""
    
    logger.warning(
        f"Validation error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Extract validation error details
    if hasattr(exc, 'errors'):
        validation_errors = exc.errors()
    else:
        validation_errors = [{"msg": str(exc)}]
    
    error_response = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid input data",
            "status_code": 422,
            "timestamp": str(datetime.now()),
            "path": request.url.path,
            "details": {
                "validation_errors": validation_errors
            }
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response
    )

async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle SQLAlchemy database exceptions."""
    
    logger.error(
        f"Database error: {type(exc).__name__}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    error_response = {
        "error": {
            "code": "DATABASE_ERROR",
            "message": "A database error occurred. Please try again later.",
            "status_code": 500,
            "timestamp": str(datetime.now()),
            "path": request.url.path
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )

# Utility Functions
def log_error(
    error: Exception,
    context: str = "Unknown",
    user_id: Optional[int] = None,
    additional_data: Optional[Dict[str, Any]] = None
):
    """
    Centralized error logging.
    
    PURPOSE:
    - Consistent error logging format
    - Include context and user information
    - Structured logging for analysis
    """
    
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "user_id": user_id
    }
    
    if additional_data:
        error_data.update(additional_data)
    
    logger.error(f"Error in {context}: {str(error)}", extra=error_data)

def create_error_response(
    message: str,
    error_code: str = "ERROR",
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create standardized error response."""
    
    from datetime import datetime
    
    error_response = {
        "error": {
            "code": error_code,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )

# Error monitoring utilities
class ErrorTracker:
    """
    Simple error tracking for monitoring.
    
    IN PRODUCTION: Replace with proper monitoring (Sentry, DataDog, etc.)
    """
    
    def __init__(self):
        self.error_counts = {}
    
    def track_error(self, error_code: str, context: str = ""):
        """Track error occurrence."""
        key = f"{error_code}:{context}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()
    
    def reset_stats(self):
        """Reset error statistics."""
        self.error_counts.clear()

# Global error tracker instance
error_tracker = ErrorTracker()

def test_exceptions():
    """Test exception handling."""
    print("🧪 TESTING EXCEPTION HANDLING:")
    print("=" * 40)
    
    # Test custom exceptions
    exceptions_to_test = [
        AuthenticationError("Invalid credentials"),
        AuthorizationError("Admin access required"),
        ValidationError("Invalid email format", field="email"),
        DocumentProcessingError("PDF extraction failed", document_id="doc_123"),
        RAGSystemError("Embedding generation failed", component="embeddings"),
        DatabaseError("Connection timeout", operation="query"),
        FileUploadError("File too large", filename="large_manual.pdf"),
        RateLimitError("Too many requests", retry_after=60)
    ]
    
    for exc in exceptions_to_test:
        print(f"\n{type(exc).__name__}:")
        print(f"  Code: {exc.error_code}")
        print(f"  Status: {exc.status_code}")
        print(f"  Message: {exc.message}")
        if exc.details:
            print(f"  Details: {exc.details}")
        
        # Track error
        error_tracker.track_error(exc.error_code, "test")
    
    # Show error statistics
    print(f"\n📊 Error Statistics:")
    stats = error_tracker.get_error_stats()
    for error, count in stats.items():
        print(f"  {error}: {count}")

if __name__ == "__main__":
    from datetime import datetime
    test_exceptions()