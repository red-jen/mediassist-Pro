"""
MEDIASSIST PRO - MAIN APPLICATION
=================================

LEARNING OBJECTIVES:
1. Understand FastAPI application structure and initialization
2. Learn middleware configuration for security and CORS
3. See how to integrate all components (DB, RAG, Auth, Routes)
4. Understand application lifecycle and startup/shutdown events

KEY CONCEPTS:
- FastAPI app initialization and configuration
- Middleware stack for cross-cutting concerns
- Dependency injection for shared resources
- Exception handling and error responses
- API documentation with OpenAPI/Swagger
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Import all our modules
from .core import (
    settings, 
    validate_environment,
    mediassist_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    database_exception_handler,
    MediAssistException
)
from .db import DatabaseUtils, get_database_session
from .rag import create_vector_store, create_rag_chain
from .api.routes import auth, query

# LLMOps — Prometheus metrics endpoint
try:
    from .monitoring.prometheus_metrics import setup_prometheus
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False

# Import for preprocessing existing documents
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import PDFProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    STARTUP TASKS:
    1. Validate configuration
    2. Initialize database
    3. Set up RAG system
    4. Process existing documents
    5. Health checks
    
    SHUTDOWN TASKS:
    1. Cleanup resources
    2. Close database connections
    3. Save final metrics
    """
    
    print("🚀 MediAssist Pro - Starting up...")
    
    # === STARTUP ===
    try:
        # 1. Validate environment and configuration
        print("🔧 Validating configuration...")
        validate_environment()
        
        # 2. Initialize database
        print("🗄️  Initializing database...")
        DatabaseUtils.init_database()
        
        # 3. Initialize RAG system
        print("🤖 Initializing RAG system...")
        vector_store = create_vector_store()
        rag_chain = create_rag_chain(vector_store)
        
        # Store in app state for access in routes
        app.state.vector_store = vector_store
        app.state.rag_chain = rag_chain
        
        # 4. Process any existing PDFs
        print("📄 Processing existing documents...")
        await process_existing_documents(vector_store)
        
        # 5. System health check
        print("🏥 Running system health check...")
        health_status = await run_health_check(rag_chain)
        
        if health_status["healthy"]:
            print("✅ System startup complete - All systems operational")
        else:
            print("⚠️  System startup complete with warnings")
            for warning in health_status.get("warnings", []):
                print(f"   - {warning}")
                
        print(f"🌐 Server will be available at: http://{settings.api_host}:{settings.api_port}")
        print(f"📚 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        raise
    
    # Application is now running
    yield
    
    # === SHUTDOWN ===
    print("🔄 MediAssist Pro - Shutting down...")
    
    try:
        # Save final metrics
        print("📊 Saving final metrics...")
        # You could save app usage statistics here
        
        # Cleanup resources
        print("🧹 Cleaning up resources...")
        # Any cleanup needed for RAG components
        
        print("✅ Shutdown complete")
        
    except Exception as e:
        print(f"⚠️  Shutdown error: {str(e)}")

async def process_existing_documents(vector_store):
    """Process any existing PDF documents in the data directory."""
    
    pdf_dir = "data"
    if not os.path.exists(pdf_dir):
        print(f"   No data directory found at {pdf_dir}")
        return
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("   No PDF files found")
        return
    
    print(f"   Found {len(pdf_files)} PDF file(s)")
    
    # Process each PDF
    processor = PDFProcessor()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            print(f"   Processing {pdf_file}...")
            chunks = processor.process_pdf_file(pdf_path)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            print(f"   ✅ Added {len(chunks)} chunks from {pdf_file}")
            
        except Exception as e:
            print(f"   ❌ Failed to process {pdf_file}: {str(e)}")

async def run_health_check(rag_chain) -> Dict[str, Any]:
    """Run comprehensive system health check."""
    
    health_status = {
        "healthy": True,
        "warnings": [],
        "components": {}
    }
    
    try:
        # Check RAG system
        system_status = rag_chain.get_system_status()
        health_status["components"]["rag"] = {
            "status": "healthy",
            "documents": system_status["vector_store"]["total_documents"]
        }
        
        if system_status["vector_store"]["total_documents"] == 0:
            health_status["warnings"].append("No documents in knowledge base")
        
        # Check database
        db_stats = DatabaseUtils.get_database_stats()
        health_status["components"]["database"] = {
            "status": "healthy",
            "users": db_stats["total_users"]
        }
        
        if db_stats["total_users"] == 0:
            health_status["warnings"].append("No users in database")
        
        # Overall health
        if health_status["warnings"]:
            health_status["healthy"] = False
            
    except Exception as e:
        health_status["healthy"] = False
        health_status["warnings"].append(f"Health check failed: {str(e)}")
    
    return health_status

# === FASTAPI APPLICATION SETUP ===

def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    CONFIGURATION INCLUDES:
    - Basic app metadata
    - CORS middleware for frontend integration
    - Security middleware
    - Exception handlers
    - Route registration
    - API documentation customization
    """
    
    # Create FastAPI app with custom configuration
    app = FastAPI(
        title="MediAssist Pro",
        description="""
        **Biomedical Equipment Technical Support RAG System**
        
        An intelligent document retrieval and question-answering system for biomedical laboratory equipment.
        
        ## Features
        
        - 🔍 **Smart Document Search**: Semantic search across technical manuals
        - 🤖 **AI-Powered Q&A**: Generate precise answers from your documentation
        - 🔐 **Secure Access**: JWT-based authentication with role management
        - 📊 **Usage Analytics**: Track query performance and user interactions
        - 📚 **Document Management**: Upload and process PDF manuals
        
        ## Getting Started
        
        1. **Authenticate**: Use `/auth/login` to get your access token
        2. **Ask Questions**: Send queries to `/query/ask` 
        3. **View History**: Check your query history at `/query/history`
        
        ## Authentication
        
        This API uses JWT Bearer tokens. Include your token in the Authorization header:
        ```
        Authorization: Bearer your_jwt_token_here
        ```
        """,
        version="1.0.0",
        contact={
            "name": "MediAssist Pro Support",
            "email": "support@mediassist.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "Authentication",
                "description": "User authentication and authorization operations"
            },
            {
                "name": "Query Processing", 
                "description": "RAG-based document search and question answering"
            },
            {
                "name": "System",
                "description": "System health and status information"
            }
        ],
        lifespan=lifespan  # Application lifecycle management
    )
    
    # === MIDDLEWARE CONFIGURATION ===
    
    # CORS Middleware (for frontend integration)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=["*"],
        expose_headers=["X-Total-Count", "X-Page-Count"]
    )
    
    # Trusted Host Middleware (security)
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=[settings.api_host, "localhost", "127.0.0.1"]
        )
    
    # === EXCEPTION HANDLERS ===
    
    # Custom exception handlers for consistent error responses
    app.add_exception_handler(MediAssistException, mediassist_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Pydantic validation errors
    from pydantic import ValidationError
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Database errors
    from sqlalchemy.exc import SQLAlchemyError
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    
    # === ROUTE REGISTRATION ===
    
    # Include API routes
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    # === PROMETHEUS METRICS ENDPOINT ===
    # Mounts /metrics (scraped by Prometheus) + HTTP latency middleware
    if PROMETHEUS_AVAILABLE:
        setup_prometheus(app)
    
    # === SYSTEM ENDPOINTS ===
    
    @app.get("/", tags=["System"])
    async def root():
        """
        Root endpoint with system information.
        
        **Returns**: Basic system info and navigation links
        """
        return {
            "name": "MediAssist Pro",
            "version": "1.0.0",
            "description": "Biomedical Equipment Technical Support RAG System",
            "status": "operational",
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "authentication": "/api/v1/auth",
                "queries": "/api/v1/query",
                "health": "/health"
            },
            "features": [
                "JWT Authentication",
                "RAG-based Q&A",
                "Document Processing",
                "Usage Analytics"
            ]
        }
    
    @app.get("/health", tags=["System"])
    async def health_check(request: Request):
        """
        Comprehensive health check endpoint.
        
        **Returns**: System health status and component information
        """
        try:
            # Check if RAG system is available
            rag_chain = getattr(request.app.state, 'rag_chain', None)
            
            if rag_chain:
                health_status = await run_health_check(rag_chain)
            else:
                health_status = {
                    "healthy": False,
                    "warnings": ["RAG system not initialized"],
                    "components": {}
                }
            
            # Add system information
            health_status.update({
                "timestamp": DatabaseUtils.get_database_stats(),
                "config": {
                    "environment": "development" if settings.debug else "production",
                    "database": "connected",
                    "version": "1.0.0"
                }
            })
            
            status_code = 200 if health_status["healthy"] else 503
            
            return JSONResponse(
                status_code=status_code,
                content=health_status
            )
            
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "healthy": False,
                    "error": str(e),
                    "timestamp": str(asyncio.get_event_loop().time())
                }
            )
    
    @app.get("/openapi.json", include_in_schema=False)
    async def custom_openapi():
        """Custom OpenAPI schema."""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="MediAssist Pro API",
            version="1.0.0", 
            description="Biomedical Equipment Technical Support RAG System",
            routes=app.routes,
        )
        
        # Customize schema
        openapi_schema["info"]["x-logo"] = {
            "url": "https://via.placeholder.com/120x60/1976D2/FFFFFF?text=MediAssist"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    return app

# Create the application instance
app = create_application()

# === DEVELOPMENT SERVER ===

def run_development_server():
    """Run development server with auto-reload."""
    
    print("🚀 Starting MediAssist Pro Development Server")
    print(f"🔧 Debug mode: {settings.debug}")
    print(f"🌐 Host: {settings.api_host}:{settings.api_port}")
    print(f"📚 Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=settings.debug
    )

if __name__ == "__main__":
    # Run development server when executing this file directly
    run_development_server()