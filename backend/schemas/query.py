# schemas/query.py
# Pydantic models for the RAG ask endpoint.

from pydantic import BaseModel
from typing import List
from datetime import datetime


# ── Request bodies ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    query:        str           # the user's question
    chat_history: List[dict] = []  # optional previous turns


# ── Response bodies ───────────────────────────────────────────────

class SourceItem(BaseModel):
    source: str
    page:   str


class AskResponse(BaseModel):
    answer:  str
    sources: List[SourceItem]


class QueryHistoryItem(BaseModel):
    id:         int
    query:      str
    reponse:    str
    created_at: datetime

    class Config:
        from_attributes = True
