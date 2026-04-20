"""
Chat routes — handles /api/chat and /api/health endpoints.
"""

from fastapi import APIRouter, HTTPException
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.llm import llm_manager
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check if the server is running and model is loaded.
    Frontend can call this on startup to verify backend is ready.
    """
    return HealthResponse(
        status="ok" if llm_manager.loaded else "model_not_loaded",
        model_loaded=llm_manager.loaded,
        model_path=settings.model_path,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts conversation history, returns AI reply.

    Body:
        {
            "messages": [
                {"role": "user", "content": "What is diabetes?"}
            ]
        }

    Response:
        {
            "reply": "Diabetes is ...",
            "role": "assistant"
        }
    """
    if not llm_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please wait and try again."
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages list cannot be empty."
        )

    # Convert Pydantic models → plain dicts for llm_manager
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        logger.info(f"Generating reply for {len(messages)} message(s)...")
        reply = llm_manager.generate(messages)
        logger.info("Reply generated successfully.")
        return ChatResponse(reply=reply)

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )