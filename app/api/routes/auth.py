"""
AUTHENTICATION ROUTES
====================

LEARNING OBJECTIVES:
1. Understand RESTful API design for authentication
2. Learn JWT token-based authentication flow
3. See request/response models with Pydantic
4. Understand proper error handling in API routes

KEY CONCEPTS:
- Login endpoint = exchange credentials for JWT token
- Token refresh = extend token validity without re-authentication
- User registration = create new user accounts
- Password management = secure password updates
"""

from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from ...db import get_database_session, User
from ...core import (
    AuthenticationService, 
    AuthorizationService, 
    PasswordValidator,
    get_current_user,
    AuthenticationError,
    ValidationError,
    settings
)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# === REQUEST/RESPONSE MODELS ===

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=1, description="Password")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "tech1",
                "password": "SecurePassword123!"
            }
        }
    }

class RegisterRequest(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    role: str = Field(default="technician", description="User role")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "new_tech",
                "email": "tech@lab.com",
                "password": "SecurePassword123!",
                "role": "technician"
            }
        }
    }

class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")
    
class UserProfile(BaseModel):
    """User profile response model."""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: datetime = None
    
    class Config:
        from_attributes = True

class RegistrationResponse(BaseModel):
    """User registration response."""
    message: str
    user_id: int
    username: str
    
# === AUTHENTICATION ENDPOINTS ===

@router.post("/login", response_model=TokenResponse, summary="User Login")
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_database_session)
):
    """
    Authenticate user and return JWT token.
    
    AUTHENTICATION FLOW:
    1. Validate credentials against database
    2. Generate JWT token with user information
    3. Return token with user profile
    4. Log successful login
    
    **Returns**: JWT access token and user information
    """
    
    try:
        # Authenticate user
        user = AuthenticationService.authenticate_user(
            db=db,
            username=login_data.username,
            password=login_data.password
        )
        
        if not user:
            raise AuthenticationError(
                message="Invalid username or password",
                details={"username": login_data.username}
            )
        
        # Create access token
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role,
            "email": user.email
        }
        
        access_token = AuthenticationService.create_access_token(
            data=token_data,
            expires_delta=timedelta(minutes=settings.jwt_expire_minutes)
        )
        
        # Prepare user info (exclude sensitive data)
        user_info = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "permissions": AuthorizationService.ROLE_PERMISSIONS.get(user.role, [])
        }
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expire_minutes * 60,  # Convert to seconds
            user_info=user_info
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )

@router.post("/register", response_model=RegistrationResponse, summary="User Registration")
async def register(
    registration_data: RegisterRequest,
    db: Session = Depends(get_database_session),
    current_user: User = Depends(get_current_user)  # Only existing users can create new users
):
    """
    Register a new user account.
    
    SECURITY CONSIDERATIONS:
    - Only authenticated users can create new accounts
    - Password validation enforced
    - Username and email uniqueness checked
    - Role assignment controlled
    
    **Requires**: Admin or technician role
    **Returns**: Registration confirmation
    """
    
    # Check if current user can create accounts
    if not AuthorizationService.has_permission(current_user, "create_users"):
        if not (current_user.is_admin or current_user.role == "technician"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create users"
            )
    
    try:
        # Validate password
        password_validation = PasswordValidator.validate_password(registration_data.password)
        if not password_validation["is_valid"]:
            raise ValidationError(
                message="Password does not meet requirements",
                details={
                    "errors": password_validation["errors"],
                    "requirements": PasswordValidator.generate_password_requirements()
                }
            )
        
        # Check username uniqueness
        existing_user = db.query(User).filter(User.username == registration_data.username).first()
        if existing_user:
            raise ValidationError(
                message="Username already exists",
                field="username"
            )
        
        # Check email uniqueness
        existing_email = db.query(User).filter(User.email == registration_data.email).first()
        if existing_email:
            raise ValidationError(
                message="Email already registered",
                field="email"
            )
        
        # Validate role
        valid_roles = ["readonly", "technician", "admin"]
        if registration_data.role not in valid_roles:
            raise ValidationError(
                message=f"Invalid role. Must be one of: {valid_roles}",
                field="role"
            )
        
        # Only admins can create admin users
        if registration_data.role == "admin" and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can create admin accounts"
            )
        
        # Create new user
        hashed_password = AuthenticationService.hash_password(registration_data.password)
        
        new_user = User(
            username=registration_data.username,
            email=registration_data.email,
            hashed_password=hashed_password,
            role=registration_data.role,
            is_active=True
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return RegistrationResponse(
            message="User registered successfully",
            user_id=new_user.id,
            username=new_user.username
        )
        
    except (ValidationError, HTTPException):
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.get("/profile", response_model=UserProfile, summary="Get User Profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get current user's profile information.
    
    **Requires**: Valid JWT token
    **Returns**: User profile data
    """
    return UserProfile.from_orm(current_user)

@router.post("/change-password", summary="Change Password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """
    Change user password.
    
    SECURITY FLOW:
    1. Verify current password
    2. Validate new password strength
    3. Update password hash in database
    4. Invalidate existing tokens (optional)
    
    **Requires**: Valid JWT token
    **Returns**: Success confirmation
    """
    
    try:
        # Verify current password
        if not AuthenticationService.verify_password(
            password_data.current_password,
            current_user.hashed_password
        ):
            raise AuthenticationError("Current password is incorrect")
        
        # Validate new password
        password_validation = PasswordValidator.validate_password(password_data.new_password)
        if not password_validation["is_valid"]:
            raise ValidationError(
                message="New password does not meet requirements",
                details={
                    "errors": password_validation["errors"],
                    "requirements": PasswordValidator.generate_password_requirements()
                }
            )
        
        # Check if new password is different
        if password_data.current_password == password_data.new_password:
            raise ValidationError("New password must be different from current password")
        
        # Update password
        new_hashed_password = AuthenticationService.hash_password(password_data.new_password)
        current_user.hashed_password = new_hashed_password
        
        db.commit()
        
        return {
            "message": "Password changed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except (AuthenticationError, ValidationError):
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )

@router.post("/logout", summary="User Logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout user (client-side token invalidation).
    
    NOTE: With JWT tokens, logout is typically handled client-side
    by removing the token. For server-side token invalidation,
    you would need a token blacklist.
    
    **Requires**: Valid JWT token
    **Returns**: Logout confirmation
    """
    
    return {
        "message": "Logged out successfully",
        "username": current_user.username,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Please remove the access token from your client"
    }

@router.get("/validate-token", summary="Validate Token")
async def validate_token(current_user: User = Depends(get_current_user)):
    """
    Validate JWT token and return user info.
    
    Useful for:
    - Frontend token validation
    - API health checks
    - User session verification
    
    **Requires**: Valid JWT token
    **Returns**: Token validation status and user info
    """
    
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role,
            "is_active": current_user.is_active
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# === UTILITY ENDPOINTS ===

@router.get("/password-requirements", summary="Get Password Requirements")
async def get_password_requirements():
    """
    Get password requirements for user registration/password change.
    
    **Returns**: Password validation rules and requirements
    """
    
    return {
        "requirements": PasswordValidator.generate_password_requirements(),
        "validation_rules": {
            "min_length": settings.security_config.PASSWORD_MIN_LENGTH,
            "require_special": settings.security_config.PASSWORD_REQUIRE_SPECIAL,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True
        },
        "strength_scoring": {
            "max_score": 100,
            "score_factors": [
                "Length (20 points base + bonuses)",
                "Lowercase letters (15 points)",
                "Uppercase letters (15 points)",
                "Numbers (15 points)",
                "Special characters (15 points)",
                "Length bonuses (10-20 points)"
            ]
        }
    }

if __name__ == "__main__":
    # Test route models
    print("🧪 TESTING AUTH ROUTE MODELS:")
    print("=" * 40)
    
    # Test request models
    login_req = LoginRequest(username="test", password="password123")
    print(f"Login Request: {login_req.model_dump()}")
    
    register_req = RegisterRequest(
        username="newuser",
        email="user@test.com",
        password="SecurePass123!",
        role="technician"
    )
    print(f"Register Request: {register_req.model_dump()}")
    
    # Test password validation
    validator = PasswordValidator()
    test_password = "TestPassword123!"
    validation = validator.validate_password(test_password)
    print(f"\nPassword validation for '{test_password}':")
    print(f"Valid: {validation['is_valid']}")
    print(f"Score: {validation['strength_score']}/100")
    
    print("\n✅ Auth route models test complete")