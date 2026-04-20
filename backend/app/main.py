"""
MedLlama Backend — FastAPI application entry point.

Startup:
    uvicorn app.main:app --reload --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import router
from app.llm import llm_manager
import logging


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Lifespan: runs on startup & shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — load the model once
    logger.info("MedLlama backend starting up...")
    llm_manager.load()
    logger.info("MedLlama is ready to answer medical questions!")
    yield
    # SHUTDOWN — cleanup (optional)
    logger.info("MedLlama shutting down.")


# App
app = FastAPI(
    title="MedLlama API",
    description="Medical AI Assistant powered by TinyLlama fine-tuned on MedQuAD",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows frontend (HTML files) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


# Root
@app.get("/")
def root():
    return {
        "name": "MedLlama API",
        "version": "1.0.0",
        "model": "TinyLlama-1.1B fine-tuned on MedQuAD",
        "method": "QLoRA",
        "docs": "/docs",
    }