"""
DATABASE PACKAGE INITIALIZATION
===============================

Provides easy access to database components.
"""

from .models import Base, User, Query, Document, SystemMetrics
from .session import get_database_session, DatabaseUtils, DatabaseSession, db_config

__all__ = [
    # Models
    "Base",
    "User", 
    "Query",
    "Document",
    "SystemMetrics",
    
    # Session management
    "get_database_session",
    "DatabaseUtils",
    "DatabaseSession",
    "db_config"
]