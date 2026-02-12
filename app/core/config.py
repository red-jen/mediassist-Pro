"""
APPLICATION CONFIGURATION
========================

LEARNING OBJECTIVES:
1. Understand centralized configuration management
2. Learn pydantic-settings for environment variable handling
3. See how to validate and type-check configuration
4. Understand different configuration for different environments

KEY CONCEPTS:
- Pydantic Settings = type-safe environment variable management
- Configuration validation = prevent runtime errors from bad config
- Environment separation = dev/test/production configurations
- Secret management = secure handling of sensitive data
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    PYDANTIC SETTINGS BENEFITS:
    - Automatic type conversion from environment variables
    - Validation of configuration values
    - Default values and documentation
    - IDE support with type hints
    """
    
    # === APPLICATION SETTINGS ===
    app_name: str = Field(default="MediAssist Pro", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # === SERVER SETTINGS ===
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # === DATABASE SETTINGS ===
    database_url: str = Field(
        default="sqlite:///./mediassist.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # === SECURITY SETTINGS ===
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=60, description="JWT token expiration (minutes)")
    
    # === RAG SYSTEM SETTINGS ===
    
    # Chunking Configuration
    chunk_size: int = Field(default=800, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        chunk_size = values.get('chunk_size', 800)
        if v >= chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    
    # Vector Store Configuration
    collection_name: str = Field(default="biomedical_manuals", description="ChromaDB collection name")
    persist_directory: str = Field(default="./data/chroma_db", description="ChromaDB persistence directory")
    
    # Retrieval Configuration
    top_k: int = Field(default=10, description="Initial retrieval count")
    final_k: int = Field(default=5, description="Final results to return to LLM")
    similarity_threshold: float = Field(default=0.6, description="Minimum similarity threshold")
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('similarity_threshold must be between 0 and 1')
        return v
    
    # LLM Configuration
    llm_model: str = Field(default="llama3.2", description="LLM model name")
    llm_temperature: float = Field(default=0.1, description="LLM temperature (0.0-1.0)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    @validator('llm_temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('llm_temperature must be between 0 and 1')
        return v
    
    # === FILE UPLOAD SETTINGS ===
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".txt", ".docx"],
        description="Allowed file extensions"
    )
    upload_directory: str = Field(default="./data/uploads", description="File upload directory")
    
    # === LOGGING SETTINGS ===
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    # === CORS SETTINGS ===
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="CORS allowed methods"
    )
    
    # === RATE LIMITING SETTINGS ===
    rate_limit_requests: int = Field(default=100, description="Rate limit: requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = ""  # No prefix, use exact names
    
    def get_database_url(self) -> str:
        """Get database URL with validation."""
        return self.database_url
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.jwt_secret_key != "your-super-secret-jwt-key-change-in-production"
    
    def validate_production_settings(self) -> List[str]:
        """Validate settings for production deployment."""
        issues = []
        
        if self.jwt_secret_key == "your-super-secret-jwt-key-change-in-production":
            issues.append("❌ JWT secret key is using default value")
        
        if len(self.jwt_secret_key) < 32:
            issues.append("❌ JWT secret key is too short (minimum 32 characters)")
        
        if self.debug:
            issues.append("⚠️  Debug mode is enabled in production")
        
        if self.database_url.startswith("sqlite://"):
            issues.append("⚠️  Using SQLite in production (consider PostgreSQL)")
        
        if self.database_echo:
            issues.append("⚠️  Database query echoing is enabled (performance impact)")
        
        return issues
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "database_type": "PostgreSQL" if "postgresql" in self.database_url else "SQLite",
            "embedding_model": self.embedding_model.split("/")[-1],  # Just model name
            "llm_model": self.llm_model,
            "collection_name": self.collection_name,
            "environment": "development" if self.is_development() else "production"
        }

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings (for dependency injection)."""
    return settings

def validate_environment():
    """Validate environment configuration and print status."""
    print("🔧 CONFIGURATION VALIDATION")
    print("=" * 50)
    
    # Print configuration summary
    summary = settings.get_summary()
    print("📊 Configuration Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Validate production settings
    if not settings.is_development():
        print("\n🔒 Production Settings Validation:")
        issues = settings.validate_production_settings()
        
        if not issues:
            print("   ✅ All production settings are valid")
        else:
            print("   Issues found:")
            for issue in issues:
                print(f"     {issue}")
    
    # Check required directories
    print("\n📁 Directory Validation:")
    directories = [
        settings.persist_directory,
        settings.upload_directory,
        os.path.dirname(settings.database_url.replace("sqlite:///", "")) if settings.database_url.startswith("sqlite:///") else None
    ]
    
    for directory in directories:
        if directory:
            if os.path.exists(directory):
                print(f"   ✅ {directory}")
            else:
                print(f"   📝 Will create: {directory}")
                os.makedirs(directory, exist_ok=True)
    
    # Check optional API keys
    print("\n🔑 API Key Status:")
    if settings.openai_api_key:
        print(f"   ✅ OpenAI API key configured")
    else:
        print(f"   ℹ️  OpenAI API key not configured (using local LLM)")
    
    print("\n✅ Configuration validation complete")

class DevelopmentSettings(Settings):
    """Development-specific settings."""
    debug: bool = True
    reload: bool = True
    database_echo: bool = True
    log_level: str = "DEBUG"
    
class ProductionSettings(Settings):
    """Production-specific settings."""
    debug: bool = False
    reload: bool = False
    database_echo: bool = False
    log_level: str = "INFO"

def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings for specific environment."""
    env = env or os.getenv("ENVIRONMENT", "development")
    
    if env.lower() == "production":
        return ProductionSettings()
    elif env.lower() == "development":
        return DevelopmentSettings()
    else:
        return Settings()

if __name__ == "__main__":
    # Test configuration
    print("🧪 TESTING CONFIGURATION:")
    print("=" * 40)
    
    # Validate current environment
    validate_environment()
    
    # Test different environments
    print(f"\n🔧 Testing different environments:")
    
    for env in ["development", "production"]:
        print(f"\n--- {env.upper()} ---")
        env_settings = get_settings_for_environment(env)
        env_summary = env_settings.get_summary()
        for key, value in env_summary.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Configuration testing complete")