"""
DATABASE MODELS - SQLAlchemy Models for MediAssist Pro
=====================================================

LEARNING OBJECTIVES:
1. Understand database design for RAG applications
2. Learn user authentication with role-based access
3. See how to log queries and responses for analytics
4. Understand relationship modeling in SQLAlchemy

KEY CONCEPTS:
- User management for secure access
- Query logging for performance monitoring
- Document tracking for audit trails
- Metadata storage for enhanced search
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """
    User model for authentication and authorization.
    
    WHY ROLE-BASED ACCESS?
    - Different users need different permissions
    - Technicians vs. Administrators vs. Read-only users
    - Audit trail for who accessed what information
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Role-based access control
    role = Column(String(20), nullable=False, default="technician")  # technician, admin, readonly
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    queries = relationship("Query", back_populates="user")
    
    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.role == "admin"
    
    @property
    def can_upload_documents(self) -> bool:
        """Check if user can upload new documents."""
        return self.role in ["admin", "technician"]

class Query(Base):
    """
    Query logging model for tracking user interactions with the RAG system.
    
    WHY LOG QUERIES?
    - Performance monitoring and optimization
    - Understanding user needs and common questions
    - Debugging and error tracking
    - Compliance and audit requirements
    """
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)
    
    # Performance metrics
    processing_time_ms = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    retrieved_chunks = Column(Integer, nullable=True)
    
    # Retrieval strategy used
    retrieval_strategy = Column(String(20), nullable=True, default="hybrid")
    
    # Sources and metadata
    sources_used = Column(JSON, nullable=True)  # List of source documents
    retrieval_metadata = Column(JSON, nullable=True)  # Additional retrieval info
    
    # User tracking
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(36), nullable=True)  # For anonymous tracking
    
    # Feedback (for continuous improvement)
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="queries")
    
    def __repr__(self):
        return f"<Query(id={self.id}, user_id={self.user_id}, confidence={self.confidence_score})>"
    
    @property
    def is_successful(self) -> bool:
        """Check if query was successfully answered (confidence > 0.5)."""
        return self.confidence_score is not None and self.confidence_score > 0.5

class Document(Base):
    """
    Document metadata model for tracking uploaded documents.
    
    WHY TRACK DOCUMENTS?
    - Version control and updates
    - Usage analytics per document
    - Quality assessment of different manuals
    - Compliance and audit trails
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Document identification
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256 for duplicate detection
    
    # Document metadata
    title = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    document_type = Column(String(50), nullable=True)  # manual, guide, protocol, etc.
    equipment_type = Column(String(100), nullable=True)  # centrifuge, spectrophotometer, etc.
    manufacturer = Column(String(100), nullable=True)
    model_number = Column(String(100), nullable=True)
    
    # Processing information
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)
    
    # Embedding information
    embedding_model = Column(String(100), nullable=True)
    vector_store_collection = Column(String(100), nullable=True)
    
    # Upload tracking
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    upload_size_bytes = Column(Integer, nullable=True)
    
    # Usage statistics
    query_count = Column(Integer, default=0)  # How many times referenced in queries
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    
    # Version control
    version = Column(String(20), default="1.0")
    is_active = Column(Boolean, default=True)
    replaced_by = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    uploader = relationship("User", foreign_keys=[uploaded_by])
    replacement = relationship("Document", remote_side=[id])
    
    def __repr__(self):
        return f"<Document(filename={self.filename}, status={self.processing_status})>"
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        return self.processing_status == "completed"
    
    @property
    def chunks_per_page(self) -> float:
        """Calculate average chunks per page (useful for quality assessment)."""
        if self.total_pages and self.total_chunks:
            return self.total_chunks / self.total_pages
        return 0.0

class SystemMetrics(Base):
    """
    System-wide metrics and performance tracking.
    
    PURPOSE:
    - Monitor system performance over time
    - Track resource usage
    - Identify optimization opportunities
    - Generate analytics reports
    """
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # ms, MB, count, etc.
    
    # Context information
    context = Column(JSON, nullable=True)  # Additional metric context
    component = Column(String(50), nullable=True)  # embeddings, retrieval, llm, etc.
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Metric({self.metric_name}={self.metric_value}{self.metric_unit or ''})>"

# Database initialization helper
def create_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

def drop_tables(engine):
    """Drop all database tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)
    print("🗑️ Database tables dropped")

# Sample data creation for testing
def create_sample_data(db_session):
    """Create sample users and data for testing."""
    
    # Import here to avoid circular imports
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Sample users with proper password hashing
    admin_user = User(
        username="admin",
        email="admin@mediassist.com",
        hashed_password=pwd_context.hash("admin123"),  # Default password: admin123
        role="admin"
    )
    
    tech_user = User(
        username="tech1",
        email="tech1@lab.com", 
        hashed_password=pwd_context.hash("tech123"),  # Default password: tech123
        role="technician"
    )
    
    readonly_user = User(
        username="readonly",
        email="readonly@lab.com",
        hashed_password=pwd_context.hash("readonly123"),  # Default password: readonly123
        role="readonly"
    )
    
    # Add users
    db_session.add(admin_user)
    db_session.add(tech_user)
    db_session.add(readonly_user)
    db_session.commit()
    
    # Sample document
    sample_doc = Document(
        filename="centrifuge_manual.pdf",
        title="Centrifuge Operation Manual",
        document_type="manual",
        equipment_type="centrifuge",
        manufacturer="LabEquip Inc",
        model_number="CE-2000",
        total_pages=45,
        total_chunks=120,
        processing_status="completed",
        uploaded_by=admin_user.id,
        upload_size_bytes=2048000
    )
    
    db_session.add(sample_doc)
    db_session.commit()
    
    print("✅ Sample data created")
    print(f"   Admin user: {admin_user.username}")
    print(f"   Tech user: {tech_user.username}")
    print(f"   Readonly user: {readonly_user.username}")
    print(f"   Sample document: {sample_doc.filename}")

if __name__ == "__main__":
    # Quick test of models
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Use SQLite for testing
    engine = create_engine("sqlite:///test_mediassist.db", echo=True)
    create_tables(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        create_sample_data(db)
        
        # Test queries
        users = db.query(User).all()
        print(f"\nCreated {len(users)} users:")
        for user in users:
            print(f"  - {user}")
            
    finally:
        db.close()
        print("\nTest completed")