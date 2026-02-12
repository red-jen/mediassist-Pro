"""
DATABASE SESSION MANAGEMENT
===========================

LEARNING OBJECTIVES:
1. Understand SQLAlchemy session management
2. Learn dependency injection in FastAPI
3. See connection pooling and transaction handling
4. Understand the database layer abstraction

KEY CONCEPTS:
- Session = database connection + transaction boundary
- Dependency injection = providing database sessions to routes
- Connection pooling = reusing database connections efficiently
- Context management = automatic cleanup of resources
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

from .models import Base

load_dotenv()

class DatabaseConfig:
    """
    Database configuration and connection management.
    
    WHY SEPARATE CONFIG CLASS?
    - Centralized database settings
    - Easy testing with different databases
    - Environment-specific configurations
    - Connection pool management
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./mediassist.db")
        self.echo = os.getenv("DEBUG", "False").lower() == "true"
        
        print(f"🔧 Database Configuration:")
        print(f"   URL: {self.database_url}")
        print(f"   Echo SQL: {self.echo}")
        
        # Configure engine based on database type
        if self.database_url.startswith("sqlite"):
            # SQLite-specific configuration
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                connect_args={"check_same_thread": False},  # Allow SQLite in threads
                poolclass=StaticPool,  # Use static pool for SQLite
            )
        else:
            # PostgreSQL or other databases
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                pool_size=5,  # Number of connections to maintain
                max_overflow=10,  # Additional connections if needed
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,  # Recycle connections every hour
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,  # Explicit transaction control
            autoflush=False,   # Manual flushing for better control
            bind=self.engine
        )
        
        print("✅ Database engine configured")
    
    def create_tables(self):
        """Create all database tables."""
        print("🔧 Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        print("✅ Database tables created")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        print("🗑️ Dropping database tables...")
        Base.metadata.drop_all(bind=self.engine)
        print("✅ Database tables dropped")
    
    def get_session(self) -> Generator[Session, None, None]:
        """
        Create a database session with automatic cleanup.
        
        CONTEXT MANAGER PATTERN:
        - Automatically closes session after use
        - Handles exceptions and rollbacks
        - Prevents connection leaks
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            print(f"❌ Database error: {e}")
            raise
        finally:
            session.close()

# Global database configuration instance
db_config = DatabaseConfig()

# FastAPI dependency function
def get_database_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    DEPENDENCY INJECTION EXPLAINED:
    - FastAPI automatically provides database sessions to routes
    - Each request gets its own session
    - Sessions are automatically cleaned up
    - Easy to test with mock databases
    
    USAGE IN ROUTES:
    @app.get("/users")
    def get_users(db: Session = Depends(get_database_session)):
        return db.query(User).all()
    """
    session = db_config.SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

# Database utilities
class DatabaseUtils:
    """Utility functions for database operations."""
    
    @staticmethod
    def init_database():
        """Initialize database with tables and sample data."""
        print("🚀 Initializing database...")
        
        # Create tables
        db_config.create_tables()
        
        # Check if we need to create sample data
        session = db_config.SessionLocal()
        try:
            from .models import User
            user_count = session.query(User).count()
            
            if user_count == 0:
                print("📝 Creating initial admin user...")
                from .models import create_sample_data
                create_sample_data(session)
            else:
                print(f"✅ Database already has {user_count} users")
                
        finally:
            session.close()
    
    @staticmethod
    def reset_database():
        """Reset database (drop and recreate all tables)."""
        print("🔄 Resetting database...")
        db_config.drop_tables()
        db_config.create_tables()
        print("✅ Database reset complete")
    
    @staticmethod
    def get_database_stats() -> dict:
        """Get database statistics."""
        session = db_config.SessionLocal()
        try:
            from .models import User, Query, Document
            
            stats = {
                "total_users": session.query(User).count(),
                "total_queries": session.query(Query).count(),
                "total_documents": session.query(Document).count(),
                "active_users": session.query(User).filter(User.is_active == True).count(),
                "processed_documents": session.query(Document).filter(Document.processing_status == "completed").count(),
                "database_url": db_config.database_url,
                "engine_info": str(db_config.engine.url)
            }
            
            return stats
            
        finally:
            session.close()

# Connection testing
def test_database_connection():
    """Test database connection and basic operations."""
    print("🧪 Testing database connection...")
    
    try:
        # Test connection
        session = db_config.SessionLocal()
        
        # Test basic query
        from .models import User
        result = session.execute("SELECT 1 as test").fetchone()
        
        if result and result.test == 1:
            print("✅ Database connection successful")
            
            # Test model operations
            user_count = session.query(User).count()
            print(f"✅ Model operations working (found {user_count} users)")
            
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
        
    finally:
        session.close()

# Context manager for manual session handling
class DatabaseSession:
    """
    Context manager for manual database session handling.
    
    USAGE:
    with DatabaseSession() as db:
        users = db.query(User).all()
        # Session automatically closed
    """
    
    def __enter__(self) -> Session:
        self.session = db_config.SessionLocal()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        self.session.close()

if __name__ == "__main__":
    # Test the database configuration
    print("🧪 TESTING DATABASE CONFIGURATION:")
    print("=" * 50)
    
    # Test connection
    if test_database_connection():
        print("\n📊 Database Stats:")
        stats = DatabaseUtils.get_database_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Test session management
    print("\n🔧 Testing session management:")
    with DatabaseSession() as db:
        # Test query (will work even with empty database)
        result = db.execute("SELECT 'Session test successful' as message").fetchone()
        print(f"   {result.message}")
    
    print("\n✅ Database configuration test complete")