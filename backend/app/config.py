from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Model
    model_path: str = "models/model-q4.gguf"
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    max_tokens: int = 512
    temperature: float = 0.7

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()