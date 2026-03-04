# tests/test_auth.py
# Tests the authentication endpoints: register and login.
#
# ── What we test ──────────────────────────────────────────────────────────────
#
#   POST /auth/register
#       ✓ New user created successfully          → 201
#       ✓ Duplicate username rejected            → 400
#       ✓ Duplicate email rejected               → 400
#       ✓ Response contains correct user fields  → id, username, email
#
#   POST /auth/login
#       ✓ Correct credentials return a JWT token → 200
#       ✓ Wrong password rejected                → 401
#       ✓ Unknown username rejected              → 401
#       ✓ Token response has the right structure → access_token, token_type


# ─── /auth/register ───────────────────────────────────────────────────────────

def test_register_success(client):
    """
    Registering a brand new user should return HTTP 201 (Created).
    """
    response = client.post("/auth/register", json={
        "username": "newdoctor",
        "email":    "newdoctor@hospital.com",
        "password": "SecurePass123!"
    })

    assert response.status_code == 201


def test_register_returns_user_fields(client):
    """
    The register response should include the user's id, username and email.
    It must NOT include the hashed password (security rule).
    """
    response = client.post("/auth/register", json={
        "username": "dr_smith",
        "email":    "smith@hospital.com",
        "password": "SecurePass123!"
    })

    body = response.json()
    assert "id"       in body
    assert "username" in body
    assert "email"    in body
    assert "hashed_password" not in body   # never expose the password hash


def test_register_duplicate_username_rejected(client):
    """
    Registering a second user with the same username should return 400.
    The endpoint must check uniqueness before inserting.
    """
    payload = {
        "username": "dr_duplicate",
        "email":    "first@hospital.com",
        "password": "Pass123!"
    }
    client.post("/auth/register", json=payload)   # first registration — succeeds

    # Second registration with same username, different email
    response = client.post("/auth/register", json={
        "username": "dr_duplicate",          # same username ← should be rejected
        "email":    "second@hospital.com",
        "password": "Pass123!"
    })

    assert response.status_code == 400
    assert "Username already taken" in response.json()["detail"]


def test_register_duplicate_email_rejected(client):
    """
    Registering with an email that already exists should return 400.
    """
    payload = {
        "username": "drfirst",
        "email":    "shared@hospital.com",
        "password": "Pass123!"
    }
    client.post("/auth/register", json=payload)   # first registration

    response = client.post("/auth/register", json={
        "username": "drsecond",
        "email":    "shared@hospital.com",   # same email ← should be rejected
        "password": "Pass123!"
    })

    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


# ─── /auth/login ──────────────────────────────────────────────────────────────

def test_login_success(client, registered_user):
    """
    Logging in with valid credentials should return HTTP 200.
    The `registered_user` fixture (from conftest.py) creates the user first.
    """
    response = client.post(
        "/auth/login",
        # Login uses form data, NOT JSON — because it uses OAuth2PasswordRequestForm
        data={
            "username": registered_user["username"],
            "password": registered_user["password"]
        }
    )

    assert response.status_code == 200


def test_login_returns_token(client, registered_user):
    """
    A successful login should return a JWT token with the correct structure.
    The token is used in the Authorization header for all protected routes.
    """
    response = client.post(
        "/auth/login",
        data={
            "username": registered_user["username"],
            "password": registered_user["password"]
        }
    )

    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    # The token itself is a non-empty string
    assert isinstance(body["access_token"], str)
    assert len(body["access_token"]) > 0


def test_login_wrong_password_rejected(client, registered_user):
    """
    Logging in with the wrong password should return 401 Unauthorized.
    """
    response = client.post(
        "/auth/login",
        data={
            "username": registered_user["username"],
            "password": "WrongPassword999!"   # not the real password
        }
    )

    assert response.status_code == 401


def test_login_unknown_user_rejected(client):
    """
    Logging in with a username that doesn't exist should return 401.
    (Same error as wrong password — we don't reveal whether the user exists)
    """
    response = client.post(
        "/auth/login",
        data={
            "username": "ghost_user",
            "password": "SomePassword123!"
        }
    )

    assert response.status_code == 401
