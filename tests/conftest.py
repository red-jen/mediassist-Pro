# tests/conftest.py
# Shared test configuration loaded automatically by pytest before any test runs.
#
# Two problems this file solves:
#
#   1. rag_engine.py loads heavy ML models (HuggingFace, CrossEncoder) at import
#      time — that would make every test take 30+ seconds and require GPU.
#      Solution: replace the entire rag_engine module with a MagicMock BEFORE
#      any backend code is imported. The mock returns fake values instantly.
#
#   2. Tests must not touch the real PostgreSQL database.
#      Solution: swap get_db (the FastAPI dependency) with one that uses an
#      in-memory SQLite database created fresh for every test function.

import sys
from unittest.mock import MagicMock

# ── Step 1: Mock rag_engine BEFORE importing anything from backend ────────────
# sys.modules is Python's module cache. By inserting a MagicMock here, any
# `import rag_engine` or `from rag_engine import ...` will get the mock instead
# of loading the real file (and its heavy ML models).
# This MUST happen before the first `from backend...` line below.
sys.modules["rag_engine"] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base, get_db

# ── Step 2: In-memory SQLite test database ────────────────────────────────────
# SQLite is a lightweight database stored entirely in RAM.
# - Starts empty before each test
# - Destroyed after each test
# - No PostgreSQL server required
#
# IMPORTANT — StaticPool:
#   Normally, SQLite :memory: creates a NEW empty database for every connection.
#   FastAPI opens a new connection per request, so the tables created by the
#   fixture would be on a different connection than the one the route uses.
#   StaticPool forces ALL connections to reuse the SAME underlying connection,
#   so the tables created in the fixture are visible to the route handler.
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # ← all connections share the same in-memory database
)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False
)


def override_get_db():
    """
    Replacement for get_db() that yields a SQLite session instead of PostgreSQL.
    FastAPI will use this during tests because we register it in dependency_overrides.
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Register the override — FastAPI swaps get_db → override_get_db in every route
app.dependency_overrides[get_db] = override_get_db


# ── Step 3: Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_database():
    """
    Runs before EVERY test function automatically (autouse=True).
    Creates all tables → runs the test → drops all tables.
    This guarantees each test starts with a completely empty database.
    """
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client():
    """
    Returns a TestClient that sends fake HTTP requests to our FastAPI app.
    No real server is started — requests are handled in-process.

    Usage in tests:
        def test_something(client):
            response = client.get("/health")
    """
    return TestClient(app)


@pytest.fixture
def registered_user(client):
    """
    Creates a real user in the test database and returns their credentials.
    Used by tests that need a user to already exist (login, auth-protected routes).
    """
    user_data = {
        "username": "testdoctor",
        "email":    "doctor@test.com",
        "password": "StrongPass123!"
    }
    client.post("/auth/register", json=user_data)
    return user_data


@pytest.fixture
def auth_token(client, registered_user):
    """
    Logs in with the registered user and returns a valid JWT Bearer token.
    Used by tests that need to access protected endpoints (/rag/ask, etc).

    Usage in tests:
        def test_protected(client, auth_token):
            response = client.post(
                "/rag/ask",
                json={...},
                headers={"Authorization": f"Bearer {auth_token}"}
            )
    """
    response = client.post(
        "/auth/login",
        # Login uses form data (OAuth2PasswordRequestForm), not JSON
        data={
            "username": registered_user["username"],
            "password": registered_user["password"]
        }
    )
    return response.json()["access_token"]
