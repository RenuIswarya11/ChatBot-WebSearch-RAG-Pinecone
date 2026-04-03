import time
from typing import Any, Callable, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import AppConfig


def call_with_retry(operation: Callable[[], Any], operation_name: str) -> Any:
    """Execute a network operation with bounded retry and backoff.

    This helper retries transient network failures (for example SSL EOF,
    temporary timeouts, and handshake interruptions) before surfacing an
    actionable error message to the caller.

    Args:
        operation (Callable[[], Any]): Zero-argument callable to execute.
        operation_name (str): Human-readable operation label for diagnostics.

    Returns:
        Any: Return value from the successful operation call.

    Raises:
        RuntimeError: Raised when all retry attempts are exhausted.
    """
    max_attempts = 4
    delay_seconds = 1.5
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2

    raise RuntimeError(
        f"{operation_name} failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    ) from last_error


def build_embeddings_with_fallback(config: AppConfig) -> Tuple[Any, int, str]:
    """Build embeddings client with Gemini-first fallback behavior.

    This function first attempts multiple Gemini embedding model IDs and
    validates each using a lightweight probe embedding call. If all Gemini
    attempts fail, it falls back to a local HuggingFace embedding model so
    indexing remains available.

    Args:
        config (AppConfig): Application configuration containing API key and
            preferred embedding models.

    Returns:
        Tuple[Any, int, str]: Embedding client, output dimension, and provider label.

    Raises:
        RuntimeError: Raised when both Gemini and HuggingFace embeddings fail.
    """
    candidate_models = [
        config.embedding_model,
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/gemini-embedding-2-preview",
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001",
        "embedding-001",
    ]
    deduped_candidates = []
    for model_name in candidate_models:
        if model_name not in deduped_candidates:
            deduped_candidates.append(model_name)

    errors = []
    for model_name in deduped_candidates:
        try:
            gemini_embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=config.gemini_api_key,
            )
            vector = gemini_embeddings.embed_query("embedding model health check")
            return gemini_embeddings, len(vector), "gemini"
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    try:
        hf_embeddings = HuggingFaceEmbeddings(model_name=config.embedding_fallback_model)
        vector = hf_embeddings.embed_query("embedding model health check")
        return hf_embeddings, len(vector), "huggingface"
    except Exception as exc:
        raise RuntimeError(
            "No supported embedding model was found. "
            "Gemini attempts failed and HuggingFace fallback also failed. "
            "Set EMBEDDING_MODEL or EMBEDDING_FALLBACK_MODEL in .env. "
            "Gemini errors: "
            + " | ".join(errors[:3])
            + f" | HuggingFace error: {exc}"
        ) from exc


def resolve_index_name_for_dimension(
    pc: Pinecone,
    base_index_name: str,
    embedding_dimension: int,
    pinecone_cloud: str,
    pinecone_region: str,
) -> str:
    """Resolve or create a Pinecone index matching embedding dimension.

    This function prevents dimension mismatch errors by checking the target
    index dimension. If mismatch is detected, it creates or reuses a
    dimension-specific suffix index and returns that name.

    Args:
        pc (Pinecone): Pinecone client instance.
        base_index_name (str): User-configured base index name.
        embedding_dimension (int): Dimension of selected embeddings.
        pinecone_cloud (str): Pinecone cloud provider for new index creation.
        pinecone_region (str): Pinecone region for new index creation.

    Returns:
        str: Resolved Pinecone index name compatible with embeddings.
    """
    existing_index_names = {item["name"] for item in call_with_retry(pc.list_indexes, "Pinecone list_indexes")}
    if base_index_name not in existing_index_names:
        call_with_retry(
            lambda: pc.create_index(
                name=base_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
            ),
            "Pinecone create_index",
        )
        return base_index_name

    description = call_with_retry(
        lambda: pc.describe_index(base_index_name),
        "Pinecone describe_index",
    )
    existing_dimension = None
    if hasattr(description, "dimension"):
        existing_dimension = int(description.dimension)
    elif isinstance(description, dict):
        existing_dimension = int(description.get("dimension", 0))
    elif hasattr(description, "to_dict"):
        existing_dimension = int(description.to_dict().get("dimension", 0))

    if existing_dimension == embedding_dimension:
        return base_index_name

    fallback_index_name = f"{base_index_name}-{embedding_dimension}".lower()
    if fallback_index_name not in existing_index_names:
        call_with_retry(
            lambda: pc.create_index(
                name=fallback_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
            ),
            "Pinecone create_index",
        )
    return fallback_index_name


def build_llm_with_fallback(config: AppConfig) -> Optional[ChatGoogleGenerativeAI]:
    """Build a Gemini chat model client with model fallback.

    This function attempts multiple Gemini model IDs and verifies each by
    running a small test invocation. It returns the first model that can
    successfully answer, reducing runtime failures from model-version
    differences across API environments.

    Args:
        config (AppConfig): Application configuration containing API key and
            preferred LLM model.

    Returns:
        Optional[ChatGoogleGenerativeAI]: A validated Gemini chat model client,
            or None when no candidate model is available.

    Raises:
        None: Errors are handled internally and None is returned on failure.
    """
    candidate_models = [
        config.llm_model,
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    deduped_candidates = []
    for model_name in candidate_models:
        if model_name not in deduped_candidates:
            deduped_candidates.append(model_name)

    errors = []
    for model_name in deduped_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=config.gemini_api_key,
            )
            llm.invoke("Health check: reply with the word 'ok'.")
            return llm
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    return None


def build_store_and_models(config: AppConfig) -> Tuple[PineconeVectorStore, Optional[ChatGoogleGenerativeAI]]:
    """Create Pinecone vector store and Gemini LLM clients.

    This function initializes embeddings and chat model clients, ensures the
    configured Pinecone index exists, and returns ready-to-use components
    for ingestion and retrieval workflows.

    Args:
        config (AppConfig): Validated runtime settings and API keys.

    Returns:
        Tuple[PineconeVectorStore, Optional[ChatGoogleGenerativeAI]]:
            Initialized vector store and chat LLM client (if available).
    """
    embeddings, embedding_dimension, _provider = build_embeddings_with_fallback(config)
    pc = Pinecone(api_key=config.pinecone_api_key)
    resolved_index_name = resolve_index_name_for_dimension(
        pc=pc,
        base_index_name=config.pinecone_index_name,
        embedding_dimension=embedding_dimension,
        pinecone_cloud=config.pinecone_cloud,
        pinecone_region=config.pinecone_region,
    )

    vector_store = PineconeVectorStore(
        index=call_with_retry(
            lambda: pc.Index(resolved_index_name),
            "Pinecone index client initialization",
        ),
        embedding=embeddings,
    )
    llm = build_llm_with_fallback(config)
    return vector_store, llm

