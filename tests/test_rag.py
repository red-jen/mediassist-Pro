# tests/test_rag.py
# Tests the RAG endpoints: /rag/ask and /rag/upload-pdf.
#
# ── Key concept: mocking ──────────────────────────────────────────────────────
#
# ask_question() calls rag_engine, DeepEval, MLflow — all require external
# services (Ollama, Chroma, Postgres). In tests we replace ask_question with
# a fake that returns a hardcoded dict instantly.
#
# unittest.mock.patch() temporarily replaces a real function with a MagicMock.
# After the `with` block ends, the real function is restored automatically.
#
#   with patch("backend.services.rag_service.ask_question") as mock_ask:
#       mock_ask.return_value = {"answer": "fake", "sources": [], "contexts": []}
#       # inside here, any call to ask_question returns the fake dict
#
# ── What we test ──────────────────────────────────────────────────────────────
#
#   POST /rag/ask
#       ✓ No token → 401 or 403 (blocked by JWT guard)
#       ✓ Valid token + mocked engine → 200 with answer
#       ✓ Response has "answer" and "sources" fields
#
#   POST /rag/upload-pdf
#       ✓ No token → 401 or 403
#       ✓ Valid token + mocked indexer → 200

from unittest.mock import patch


# ─── /rag/ask ─────────────────────────────────────────────────────────────────

def test_ask_without_token_rejected(client):
    """
    Calling /rag/ask without a Bearer token should be blocked.
    The JWT guard (get_current_user dependency) should return 401 or 403.
    """
    response = client.post(
        "/rag/ask",
        json={"query": "What is ELISA?", "chat_history": []}
    )

    # Either 401 (Unauthorized) or 403 (Forbidden) — both mean "no access"
    assert response.status_code in (401, 403)


def test_ask_with_valid_token_returns_200(client, auth_token):
    """
    Calling /rag/ask with a valid token should return HTTP 200.
    We mock ask_question so no real ML inference happens.
    """
    with patch("backend.routers.rag.ask_question") as mock_ask:
        # Define what the fake ask_question returns
        mock_ask.return_value = {
            "answer":   "ELISA is an enzyme-linked immunosorbent assay.",
            "sources":  [{"source": "manual.pdf", "page": "5"}],
            "contexts": ["ELISA plates must be washed 3 times with PBS buffer."]
        }

        response = client.post(
            "/rag/ask",
            json={"query": "What is ELISA?", "chat_history": []},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    assert response.status_code == 200


def test_ask_response_has_required_fields(client, auth_token):
    """
    The response body must contain "answer" and "sources".
    These are defined in AskResponse schema (schemas/query.py).
    """
    with patch("backend.routers.rag.ask_question") as mock_ask:
        mock_ask.return_value = {
            "answer":   "ELISA is used for protein detection.",
            "sources":  [{"source": "protocol.pdf", "page": "12"}],
            "contexts": ["ELISA is used to detect specific proteins."]
        }

        response = client.post(
            "/rag/ask",
            json={"query": "What is ELISA used for?", "chat_history": []},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    body = response.json()
    assert "answer"  in body
    assert "sources" in body
    # answer should be a non-empty string
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0


def test_ask_with_chat_history(client, auth_token):
    """
    The endpoint should accept a non-empty chat_history list without errors.
    This tests the multi-turn conversation feature.
    """
    with patch("backend.routers.rag.ask_question") as mock_ask:
        mock_ask.return_value = {
            "answer":   "The second wash step lasts 3 minutes.",
            "sources":  [],
            "contexts": []
        }

        response = client.post(
            "/rag/ask",
            json={
                "query": "How long does the second step take?",
                "chat_history": [
                    {
                        "question": "What is the washing protocol?",
                        "answer":   "It involves 3 wash steps with PBS buffer."
                    }
                ]
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    assert response.status_code == 200


# ─── /rag/upload-pdf ──────────────────────────────────────────────────────────

def test_upload_pdf_without_token_rejected(client):
    """
    Uploading a PDF without a token should be blocked by the JWT guard.
    """
    response = client.post(
        "/rag/upload-pdf",
        files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    )

    assert response.status_code in (401, 403)


def test_upload_pdf_with_valid_token_returns_200(client, auth_token):
    """
    Uploading a PDF with a valid token should return 200.
    We mock index_document so no real ChromaDB/embedding happens.
    """
    with patch("backend.routers.rag.index_document") as mock_index:
        mock_index.return_value = {
            "status":  "indexed",
            "chunks":  5,
            "message": "PDF indexed successfully"
        }

        response = client.post(
            "/rag/upload-pdf",
            files={"file": ("protocol.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    assert response.status_code == 200
