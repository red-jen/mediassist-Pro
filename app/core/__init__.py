"""
CORE PACKAGE INITIALIZATION
===========================

Provides access to core application components.
"""

from .config import settings, get_settings, validate_environment
from .security import (
    AuthenticationService, 
    AuthorizationService, 
    PasswordValidator,
    get_current_user, 
    require_admin_user,
    security_config
)
from .exceptions import (
    MediAssistException,
    AuthenticationError,
    AuthorizationError, 
    ValidationError,
    DocumentProcessingError,
    RAGSystemError,
    DatabaseError,
    FileUploadError,
    RateLimitError,
    error_tracker,
    mediassist_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    database_exception_handler
)

__all__ = [
    # Configuration
    "settings",
    "get_settings", 
    "validate_environment",
    
    # Security
    "AuthenticationService",
    "AuthorizationService",
    "PasswordValidator",
    "get_current_user",
    "require_admin_user", 
    "security_config",
    
    # Exceptions
    "MediAssistException",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError", 
    "DocumentProcessingError",
    "RAGSystemError",
    "DatabaseError",
    "FileUploadError",
    "RateLimitError",
    "error_tracker",
    "mediassist_exception_handler",
    "general_exception_handler",
    "validation_exception_handler",
    "database_exception_handler"
]