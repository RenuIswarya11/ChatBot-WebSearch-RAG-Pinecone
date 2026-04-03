import time
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.config import AppConfig


def validate_uploads(uploaded_files: List[UploadedFile], config: AppConfig) -> Tuple[bool, str]:
    """Validate uploaded PDF count and page-length constraints.

    This function enforces assessment constraints: maximum number of files
    and maximum pages per file. It returns a single validation result used
    by the UI to gate indexing.

    Args:
        uploaded_files (List[UploadedFile]): Uploaded PDF files from Streamlit.
        config (AppConfig): Runtime limits for files and pages.

    Returns:
        Tuple[bool, str]: Validation status and user-facing message.
    """
    if len(uploaded_files) > config.max_files:
        return False, f"Please upload up to {config.max_files} files."

    for upload in uploaded_files:
        try:
            reader = PdfReader(upload)
            page_count = len(reader.pages)
            if page_count > config.max_pages_per_file:
                return (
                    False,
                    f"`{upload.name}` has {page_count} pages. Max allowed is {config.max_pages_per_file}.",
                )
        except PdfReadError:
            return (
                False,
                f"`{upload.name}` is not a valid or readable PDF. Please upload a different file.",
            )
        except Exception:
            return (
                False,
                f"`{upload.name}` could not be processed. Please verify it is a proper PDF.",
            )
        finally:
            upload.seek(0)
    return True, "Files are valid and ready for indexing."


def infer_section_title(page_text: str, page_number: int) -> str:
    """Infer a concise section title from page text lines.

    The first few non-empty lines are scanned to produce a readable heading
    for citation formatting. If no suitable heading exists, the page number
    fallback is used.

    Args:
        page_text (str): Extracted page-level text content.
        page_number (int): One-indexed page number for fallback labeling.

    Returns:
        str: Human-friendly section title for citations.
    """
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:8]:
        if 4 <= len(line) <= 90:
            return line
    return f"Page {page_number}"


def parse_pdfs_to_documents(uploaded_files: List[UploadedFile]) -> List[Document]:
    """Parse uploaded PDFs into page-level LangChain documents.

    Each extracted page receives metadata needed for later citation display,
    including document name, page number, and inferred section title.

    Args:
        uploaded_files (List[UploadedFile]): Uploaded PDF files to parse.

    Returns:
        List[Document]: Parsed page-level documents with metadata.
    """
    documents: List[Document] = []
    for upload in uploaded_files:
        try:
            reader = PdfReader(upload)
        except Exception:
            upload.seek(0)
            continue

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            page_number = page_idx + 1
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "doc_name": upload.name,
                        "page_number": page_number,
                        "section_title": infer_section_title(page_text, page_number),
                    },
                )
            )
        upload.seek(0)
    return documents


def chunk_documents(documents: List[Document], config: AppConfig) -> List[Document]:
    """Split page-level documents into retrieval-ready chunks.

    This function applies recursive splitting with configurable chunk size
    and overlap, preserving metadata for downstream citation rendering.

    Args:
        documents (List[Document]): Parsed page-level documents.
        config (AppConfig): Chunking configuration values.

    Returns:
        List[Document]: Chunked documents with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return chunks


def upsert_chunks(vector_store: PineconeVectorStore, chunks: List[Document], namespace: str) -> int:
    """Upsert chunked documents into Pinecone by session namespace.

    Generated IDs remain deterministic within a session namespace, enabling
    isolated retrieval for each user session.

    Args:
        vector_store (PineconeVectorStore): Vector store target for upserts.
        chunks (List[Document]): Chunked documents to index.
        namespace (str): Session-specific namespace key.

    Returns:
        int: Number of chunks successfully submitted for indexing.
    """
    if not chunks:
        return 0
    ids = [f"{namespace}-{idx}" for idx in range(len(chunks))]
    max_attempts = 4
    delay_seconds = 1.5
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            vector_store.add_documents(documents=chunks, ids=ids, namespace=namespace)
            return len(chunks)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2
    raise RuntimeError(
        f"Pinecone upsert failed after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error

