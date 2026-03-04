# models/query.py
# SQLAlchemy model for the `Query` table.
# Stores every question asked + the RAG answer generated.

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from backend.database import Base


class Query(Base):
    __tablename__ = "queries"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    query      = Column(Text, nullable=False)    # the user's question
    reponse    = Column(Text, nullable=False)    # the LLM's answer
    created_at = Column(DateTime(timezone=True), server_default=func.now())
