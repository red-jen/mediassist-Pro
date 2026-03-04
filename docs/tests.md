# Unit Tests — How They Work

## The Problem

Running real tests against a FastAPI app needs:
- A running PostgreSQL database
- Loaded ML models (HuggingFace, CrossEncoder — takes 30+ seconds)
- A live Ollama server

That's too heavy for tests that should run in seconds. The solution is to **replace the slow/external things with fakes**.

---

## Two Core Tricks

### 1. Mock `rag_engine` before it loads

`rag_engine.py` loads 2 GB of ML models the moment it is imported. We block that by injecting a fake into Python's module cache **before** any backend code is imported.

```python
# conftest.py — must be the very first thing
import sys
from unittest.mock import MagicMock

sys.modules["rag_engine"] = MagicMock()
# Now any `import rag_engine` returns a MagicMock instantly
# No models loaded, no GPU needed, no 30-second wait
```

### 2. Swap PostgreSQL with SQLite in-memory

```
Real app:                    Tests:
  FastAPI route               FastAPI route (same code)
      ↓                             ↓
  PostgreSQL           →      SQLite :memory: (lives in RAM)
  (external server)           (created fresh per test, 0 setup)
```

This is done via FastAPI's `dependency_overrides` — it tells the app to use our test database session instead of the real `get_db`:

```python
app.dependency_overrides[get_db] = override_get_db
# Every route that has `db: Session = Depends(get_db)`
# now gets a SQLite session instead of PostgreSQL
```

**Why `StaticPool`?**
SQLite `:memory:` normally creates a NEW empty database for every connection. FastAPI opens connections per request, so the tables created by the fixture would be invisible to the route. `StaticPool` forces all connections to share the same in-memory database:

```python
create_engine("sqlite:///:memory:", poolclass=StaticPool)
```

---

## File Structure

```
tests/
  conftest.py      ← shared setup loaded automatically before every test
  test_health.py   ← tests for / and /health
  test_auth.py     ← tests for /auth/register and /auth/login
  test_rag.py      ← tests for /rag/ask and /rag/upload-pdf
  __init__.py
pytest.ini         ← pytest configuration (verbosity, test paths)
```

---

## conftest.py — The Setup File

pytest loads `conftest.py` automatically before running any test. It defines 3 fixtures that every test can use:

```
conftest.py
  │
  ├── reset_database (autouse=True)
  │     Runs before EVERY test:
  │       1. Create all tables in SQLite
  │       2. Run the test
  │       3. Drop all tables
  │     → Each test starts with a completely empty database
  │
  ├── client
  │     Returns a TestClient — simulates HTTP requests without a real server
  │     Usage: def test_something(client):
  │                response = client.get("/health")
  │
  ├── registered_user
  │     Calls POST /auth/register → creates a real user in the test DB
  │     Returns the credentials dict (username, email, password)
  │     Usage: def test_login(client, registered_user):
  │
  └── auth_token
        Calls POST /auth/login with registered_user credentials
        Returns a valid JWT Bearer token string
        Usage: def test_protected_route(client, auth_token):
                   headers = {"Authorization": f"Bearer {auth_token}"}
```

---

## How Each Test File Works

### test_health.py — 4 tests

These are the simplest — no database, no auth, just HTTP GET:

```python
def test_health_returns_ok_status(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
```

```
TestClient.get("/health")
    → FastAPI handles request in-process
    → health() function runs
    → returns {"status": "ok"}
assert passes ✓
```

---

### test_auth.py — 8 tests

Tests register and login. These DO hit the (SQLite) database:

```python
def test_register_success(client):
    response = client.post("/auth/register", json={
        "username": "newdoctor",
        "email":    "newdoctor@hospital.com",
        "password": "SecurePass123!"
    })
    assert response.status_code == 201
```

```
POST /auth/register
    → register() in auth.py runs
    → queries SQLite for duplicate username/email (empty → no duplicate)
    → hashes password with bcrypt
    → inserts new User row into SQLite
    → returns 201 + user JSON
assert passes ✓
```

**Login uses form data — not JSON:**
```python
# WRONG — login uses OAuth2PasswordRequestForm, not JSON body
client.post("/auth/login", json={"username": ..., "password": ...})

# CORRECT
client.post("/auth/login", data={"username": ..., "password": ...})
```

---

### test_rag.py — 6 tests

RAG endpoints call `ask_question()` which triggers ML inference, MLflow, DeepEval. We mock that with `patch()`:

```python
from unittest.mock import patch

def test_ask_with_valid_token_returns_200(client, auth_token):
    with patch("backend.routers.rag.ask_question") as mock_ask:
        # Define what ask_question will return (instantly, no ML)
        mock_ask.return_value = {
            "answer":   "ELISA is an enzyme-linked immunosorbent assay.",
            "sources":  [{"source": "manual.pdf", "page": "5"}],
            "contexts": ["ELISA plates must be washed 3 times."]
        }

        response = client.post(
            "/rag/ask",
            json={"query": "What is ELISA?", "chat_history": []},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    assert response.status_code == 200
```

**Why patch at `backend.routers.rag.ask_question` and not at `backend.services.rag_service.ask_question`?**

Because the router does `from backend.services.rag_service import ask_question`, which creates a **local copy** of the reference inside the router module. Patching the original service doesn't affect the copy the router already holds. You must patch where it is **used**:

```
# The router file has:
from backend.services.rag_service import ask_question
#                                        ↑ this is now a local name in rag.py

# So we patch the local name:
patch("backend.routers.rag.ask_question")    ← CORRECT
patch("backend.services.rag_service.ask_question")  ← doesn't affect the router
```

---

## Running the Tests

```powershell
# Run all tests
.\.venv\Scripts\pytest

# Run a specific file
.\.venv\Scripts\pytest tests/test_auth.py

# Run a specific test
.\.venv\Scripts\pytest tests/test_auth.py::test_login_success

# Show coverage report
.\.venv\Scripts\pytest --cov=backend
```

Expected output:
```
tests/test_auth.py::test_register_success            PASSED
tests/test_auth.py::test_register_returns_user_fields PASSED
tests/test_auth.py::test_register_duplicate_username_rejected PASSED
tests/test_auth.py::test_register_duplicate_email_rejected PASSED
tests/test_auth.py::test_login_success               PASSED
tests/test_auth.py::test_login_returns_token         PASSED
tests/test_auth.py::test_login_wrong_password_rejected PASSED
tests/test_auth.py::test_login_unknown_user_rejected  PASSED
tests/test_health.py::test_root_returns_200          PASSED
tests/test_health.py::test_root_returns_message      PASSED
tests/test_health.py::test_health_returns_200        PASSED
tests/test_health.py::test_health_returns_ok_status  PASSED
tests/test_rag.py::test_ask_without_token_rejected   PASSED
tests/test_rag.py::test_ask_with_valid_token_returns_200 PASSED
tests/test_rag.py::test_ask_response_has_required_fields PASSED
tests/test_rag.py::test_ask_with_chat_history        PASSED
tests/test_rag.py::test_upload_pdf_without_token_rejected PASSED
tests/test_rag.py::test_upload_pdf_with_valid_token_returns_200 PASSED

18 passed in 4.13s
```
