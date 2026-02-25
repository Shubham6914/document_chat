"""
Configuration settings for the Legal Document Chat application.
Loads environment variables and provides centralized configuration management.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    requesty_api_key: str = Field(..., env="REQUESTY_API_KEY")
    requesty_base_url: str = Field(
        default="https://router.requesty.ai/v1",
        env="REQUESTY_BASE_URL"
    )
    
    # Model Configuration (use provider/model format for Requesty)
    model_name: str = Field(default="openai/gpt-4o-mini", env="MODEL_NAME")
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(
        default=1024,
        env="EMBEDDING_DIMENSION"
    )
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(
        default="legal-documents",
        env="PINECONE_INDEX_NAME"
    )
    
    # Vector Database Configuration
    collection_name: str = Field(
        default="legal_documents",
        env="COLLECTION_NAME"
    )
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Application Configuration
    app_title: str = Field(
        default="Legal Document Chat Assistant",
        env="APP_TITLE"
    )
    app_icon: str = Field(default="⚖️", env="APP_ICON")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    
    # Extraction Configuration
    extraction_temperature: float = Field(
        default=0.0,
        env="EXTRACTION_TEMPERATURE"
    )
    extraction_max_retries: int = Field(
        default=3,
        env="EXTRACTION_MAX_RETRIES"
    )
    confidence_threshold: float = Field(
        default=0.7,
        env="CONFIDENCE_THRESHOLD"
    )
    
    # Chat Configuration
    chat_history_length: int = Field(default=10, env="CHAT_HISTORY_LENGTH")
    max_context_chunks: int = Field(default=5, env="MAX_CONTEXT_CHUNKS")
    citation_format: str = Field(default="detailed", env="CITATION_FORMAT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for directory in [DATA_DIR, SAMPLE_DOCS_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Singleton instance
settings = get_settings()
