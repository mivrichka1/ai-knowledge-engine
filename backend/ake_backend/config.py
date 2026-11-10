from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://ake:ake@localhost:5432/ake"
    jwt_secret: str = Field(min_length=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    document_storage_path: str = "./storage"
    max_chunks_per_document: int = Field(default=5000, ge=1, le=50000)
    max_upload_size: int = Field(default=10_000_000, gt=0, le=100_000_000)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    chunk_size: int = Field(default=120, ge=1, le=4096)
    chunk_overlap: int = Field(default=16, ge=0)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    internal_api_url: str = "http://backend:8000/api/v1"
    access_minutes: int = Field(default=15, ge=1, le=1440)
    refresh_days: int = Field(default=7, ge=1, le=90)

    @model_validator(mode="after")
    def validate_chunk_configuration(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
