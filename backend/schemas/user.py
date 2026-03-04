# schemas/user.py
# Pydantic models for request validation and response serialization.
# These define the SHAPE of data coming in and going out of the API.

from pydantic import BaseModel, EmailStr


# ── Request bodies (what the client sends) ──────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    EmailStr   # pydantic validates it's a valid email format
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Response bodies (what the API returns) ────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class UserResponse(BaseModel):
    id:       int
    username: str
    email:    str
    role:     str

    class Config:
        from_attributes = True  # allows building from SQLAlchemy model instances
