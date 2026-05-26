"""
Pydantic schemas for the API layer.
Defines strict validation for incoming requests and outgoing responses.
"""
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""
    message: str
    thread_id: str | None = None

class ChatResponse(BaseModel):
    """Schema for outgoing chat responses."""
    response: str
    thread_id: str
    agents_used: list[str]
    confidence: float | None = None
