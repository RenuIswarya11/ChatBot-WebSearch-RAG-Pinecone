import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Store validated application configuration values."""

    gemini_api_key: str
    pinecone_api_key: str
    serpapi_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    llm_model: str
    embedding_model: str
    embedding_fallback_model: str
    chunk_size: int
    chunk_overlap: int
    max_files: int
    max_pages_per_file: int
    min_relevance_threshold: float
    min_keyword_overlap: float


def load_app_config() -> AppConfig:
    """Load and validate app configuration from environment variables.

    This function reads required and optional environment variables,
    enforces mandatory key presence, and returns a strongly typed
    `AppConfig` instance used across services and UI.

    Args:
        None: Configuration is read from process environment variables.

    Returns:
        AppConfig: Validated application configuration object.

    Raises:
        RuntimeError: Raised when any required API key is missing.
    """
    load_dotenv()
    required_keys = ["GEMINI_API_KEY", "PINECONE_API_KEY", "SERPAPI_API_KEY"]
    env_map: Dict[str, str] = {}
    missing = []
    for key in required_keys:
        value = os.getenv(key, "").strip()
        env_map[key] = value
        if not value:
            missing.append(key)

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them in your .env or deployment secrets."
        )

    return AppConfig(
        gemini_api_key=env_map["GEMINI_API_KEY"],
        pinecone_api_key=env_map["PINECONE_API_KEY"],
        serpapi_api_key=env_map["SERPAPI_API_KEY"],
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "smart-research-assistant"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        embedding_fallback_model=os.getenv(
            "EMBEDDING_FALLBACK_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        max_files=int(os.getenv("MAX_FILES", "5")),
        max_pages_per_file=int(os.getenv("MAX_PAGES_PER_FILE", "10")),
        min_relevance_threshold=float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.30")),
        min_keyword_overlap=float(os.getenv("MIN_KEYWORD_OVERLAP", "0.12")),
    )

