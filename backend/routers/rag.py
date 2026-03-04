import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.user import User
from backend.schemas.query import AskRequest, AskResponse
from backend.services.rag_service import ask_question, index_document
from backend.dependencies import get_current_user

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db:      Session = Depends(get_db),
    user:    User    = Depends(get_current_user)             
):
    """
    Ask a question against the indexed documents.
    Requires a valid Bearer token.
    Saves the query + response to the database.
    """
    result = ask_question(
        query        = request.query,
        chat_history = request.chat_history,
        user_id      = user.id,
        db           = db
    )
    return result

@router.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...),
    user: User       = Depends(get_current_user)             
):
    """
    Upload a PDF and index it into ChromaDB + BM25.
    Requires a valid Bearer token.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    result = index_document(tmp_path)
    os.unlink(tmp_path)                      
    return result
