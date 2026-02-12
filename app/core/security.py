"""
AUTHENTICATION & SECURITY MODULE
===============================

LEARNING OBJECTIVES:
1. Understand JWT-based authentication in FastAPI
2. Learn password hashing and verification best practices
3. See role-based access control implementation
4. Understand security headers and middleware

KEY CONCEPTS:
- JWT = JSON Web Tokens for stateless authentication
- Bcrypt = secure password hashing algorithm
- RBAC = Role-Based Access Control
- Security middleware = automatic security enforcement
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from ..db import get_database_session, User

load_dotenv()

class SecurityConfig:
    """
    Centralized security configuration.
    
    WHY CENTRALIZE SECURITY CONFIG?
    - Single source of truth for security settings
    - Easy to update security parameters
    - Environment-specific configurations
    - Consistent security across the application
    """
    
    # JWT Configuration
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    
    # Password Configuration
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_SPECIAL = True
    
    # Rate limiting (requests per minute)
    LOGIN_RATE_LIMIT = 5
    API_RATE_LIMIT = 60
    
    @classmethod
    def validate_config(cls):
        """Validate security configuration."""
        if cls.SECRET_KEY == "your-super-secret-jwt-key-change-in-production":
            print("⚠️  WARNING: Using default JWT secret key! Change in production!")
        
        if len(cls.SECRET_KEY) < 32:
            print("⚠️  WARNING: JWT secret key is too short! Use at least 32 characters!")

# Initialize security configuration
security_config = SecurityConfig()
security_config.validate_config()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security_scheme = HTTPBearer()

class PasswordValidator:
    """
    Password validation utilities.
    
    PASSWORD SECURITY PRINCIPLES:
    - Minimum length for brute-force resistance
    - Character variety for increased entropy
    - Common password detection
    - Clear feedback for users
    """
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """
        Validate password strength and return detailed feedback.
        
        RETURNS:
        - is_valid: bool
        - errors: list of validation errors
        - strength_score: 0-100 password strength score
        """
        errors = []
        strength_score = 0
        
        # Length check
        if len(password) < security_config.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {security_config.PASSWORD_MIN_LENGTH} characters long")
        else:
            strength_score += 20
        
        # Character variety checks
        if not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letters")
        else:
            strength_score += 15
            
        if not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letters")
        else:
            strength_score += 15
            
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain numbers")
        else:
            strength_score += 15
            
        if security_config.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain special characters (!@#$%^&* etc.)")
            else:
                strength_score += 15
        
        # Common password check (basic)
        common_passwords = ["password", "123456", "admin", "letmein", "welcome"]
        if password.lower() in common_passwords:
            errors.append("Password is too common, please choose a different one")
            strength_score = 0
        
        # Length bonus
        if len(password) >= 12:
            strength_score += 10
        if len(password) >= 16:
            strength_score += 10
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "strength_score": min(100, strength_score)
        }
    
    @staticmethod
    def generate_password_requirements() -> str:
        """Generate human-readable password requirements."""
        requirements = [
            f"At least {security_config.PASSWORD_MIN_LENGTH} characters long",
            "Contains lowercase letters (a-z)",
            "Contains uppercase letters (A-Z)",
            "Contains numbers (0-9)"
        ]
        
        if security_config.PASSWORD_REQUIRE_SPECIAL:
            requirements.append("Contains special characters (!@#$%^&* etc.)")
        
        return "Password requirements:\n" + "\n".join(f"• {req}" for req in requirements)

class AuthenticationService:
    """
    Core authentication service with JWT token management.
    
    JWT AUTHENTICATION FLOW:
    1. User provides username/password
    2. Server verifies credentials
    3. Server generates JWT token with user info
    4. Client includes token in requests
    5. Server validates token on each request
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using bcrypt.
        
        BCRYPT ADVANTAGES:
        - Adaptive hashing (adjustable work factor)
        - Built-in salt generation
        - Slow by design (resistant to brute force)
        - Industry standard for password hashing
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        HANDLES BOTH:
        - Passlib bcrypt hashes (standard)
        - Direct bcrypt hashes (from reset script)
        """
        try:
            # First try passlib verification
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # Fallback to direct bcrypt verification (for reset script passwords)
            try:
                import bcrypt
                return bcrypt.checkpw(
                    plain_password.encode('utf-8'), 
                    hashed_password.encode('utf-8')
                )
            except Exception:
                return False
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token.
        
        TOKEN PAYLOAD:
        - sub: subject (user identifier)
        - exp: expiration time
        - iat: issued at time
        - custom claims: role, permissions, etc.
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=security_config.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access_token"
        })
        
        encoded_jwt = jwt.encode(to_encode, security_config.SECRET_KEY, algorithm=security_config.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        RAISES:
        - JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, security_config.SECRET_KEY, algorithms=[security_config.ALGORITHM])
            
            # Validate token type
            if payload.get("type") != "access_token":
                raise JWTError("Invalid token type")
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        SECURITY CONSIDERATIONS:
        - Constant-time comparison to prevent timing attacks
        - Account lockout after failed attempts (TODO)
        - Logging of authentication attempts (TODO)
        """
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return None
        
        if not user.is_active:
            return None
            
        if not AuthenticationService.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user

class AuthorizationService:
    """
    Role-based access control and permission management.
    
    RBAC CONCEPTS:
    - Roles: admin, technician, readonly
    - Permissions: read, write, upload, admin
    - Resource-based access: own queries vs. all queries
    """
    
    # Role hierarchy (higher number = more permissions)
    ROLE_HIERARCHY = {
        "readonly": 1,
        "technician": 2, 
        "admin": 3
    }
    
    # Role permissions
    ROLE_PERMISSIONS = {
        "readonly": ["read_own_queries", "search_documents"],
        "technician": ["read_own_queries", "search_documents", "create_queries", "rate_responses"],
        "admin": ["*"]  # All permissions
    }
    
    @classmethod
    def has_permission(cls, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        if not user or not user.is_active:
            return False
        
        user_permissions = cls.ROLE_PERMISSIONS.get(user.role, [])
        
        # Admin has all permissions
        if "*" in user_permissions:
            return True
        
        return permission in user_permissions
    
    @classmethod
    def requires_role(cls, required_role: str):
        """Decorator to require specific role level."""
        def decorator(func):
            def wrapper(current_user: User, *args, **kwargs):
                user_level = cls.ROLE_HIERARCHY.get(current_user.role, 0)
                required_level = cls.ROLE_HIERARCHY.get(required_role, 999)
                
                if user_level < required_level:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required role: {required_role}"
                    )
                
                return func(current_user, *args, **kwargs)
            return wrapper
        return decorator

# FastAPI Dependencies for Authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_database_session)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    
    AUTHENTICATION FLOW:
    1. Extract Bearer token from Authorization header
    2. Verify and decode JWT token
    3. Get user from database using token subject
    4. Return authenticated user object
    
    USAGE IN ROUTES:
    @app.get("/protected")
    def protected_route(current_user: User = Depends(get_current_user)):
        return {"message": f"Hello, {current_user.username}!"}
    """
    
    # Verify token
    payload = AuthenticationService.verify_token(credentials.credentials)
    username: str = payload.get("sub")
    
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = db.query(User).filter(User.username == username).first()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def require_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency that requires admin role."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Security utilities
def generate_session_id() -> str:
    """Generate secure session ID."""
    import secrets
    return secrets.token_urlsafe(32)

def sanitize_input(input_str: str) -> str:
    """Basic input sanitization."""
    if not input_str:
        return ""
    
    # Remove potential injection characters
    sanitized = input_str.replace("<", "&lt;").replace(">", "&gt;")
    sanitized = sanitized.replace("'", "&#x27;").replace('"', "&quot;")
    
    return sanitized.strip()

# Test functions
def test_authentication():
    """Test authentication functionality."""
    print("🧪 TESTING AUTHENTICATION:")
    print("=" * 40)
    
    # Test password validation
    test_passwords = ["weak", "StrongPass123!", "admin", "MySecure@Password2024"]
    
    for password in test_passwords:
        result = PasswordValidator.validate_password(password)
        print(f"\nPassword: '{password}'")
        print(f"Valid: {result['is_valid']}")
        print(f"Strength: {result['strength_score']}/100")
        if result['errors']:
            print(f"Errors: {result['errors']}")
    
    # Test password hashing
    print(f"\n🔐 Testing password hashing:")
    test_password = "TestPassword123!"
    hashed = AuthenticationService.hash_password(test_password)
    print(f"Original: {test_password}")
    print(f"Hashed: {hashed[:50]}...")
    print(f"Verification: {AuthenticationService.verify_password(test_password, hashed)}")
    
    # Test JWT token
    print(f"\n🎫 Testing JWT tokens:")
    token_data = {"sub": "testuser", "role": "technician"}
    token = AuthenticationService.create_access_token(token_data)
    print(f"Token: {token[:50]}...")
    
    try:
        decoded = AuthenticationService.verify_token(token)
        print(f"Decoded: {decoded}")
    except Exception as e:
        print(f"Token verification failed: {e}")

if __name__ == "__main__":
    test_authentication()