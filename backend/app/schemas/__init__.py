from pydantic import BaseModel
from typing import List, Literal


class Message(BaseModel):
    """Single message in a conversation."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint."""
    messages: List[Message]

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "What are the symptoms of diabetes?"}
                ]
            }
        }
    }


class ChatResponse(BaseModel):
    """Response body from /api/chat endpoint."""
    reply: str
    role: str = "assistant"


class HealthResponse(BaseModel):
    """Response body from /api/health endpoint."""
    status: str
    model_loaded: bool
    model_path: str