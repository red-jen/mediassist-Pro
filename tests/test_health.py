# tests/test_health.py
# Tests the two basic informational endpoints.
# These are the simplest tests — no database, no auth, just HTTP GET.

# ── What we test ──────────────────────────────────────────────────────────────
#
#   GET /        → {"message": "MediAssist Pro API is running"}
#   GET /health  → {"status": "ok"}
#
# Both should always return 200 as long as the server is running.
# The /health endpoint is especially important — Docker uses it to decide
# whether to restart the container.


def test_root_returns_200(client):
    """
    Calling GET / should return HTTP 200.
    """
    response = client.get("/")

    assert response.status_code == 200


def test_root_returns_message(client):
    """
    GET / should return a JSON body with a "message" key.
    """
    response = client.get("/")

    assert "message" in response.json()


def test_health_returns_200(client):
    """
    GET /health should return HTTP 200.
    Docker HEALTHCHECK relies on this — if it returns anything else,
    Docker will mark the container as unhealthy and restart it.
    """
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_ok_status(client):
    """
    GET /health should return {"status": "ok"}.
    """
    response = client.get("/health")

    assert response.json() == {"status": "ok"}
