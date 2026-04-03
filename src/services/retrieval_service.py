import re
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.models import RetrievedContext


def extract_keywords(text: str) -> Set[str]:
    """Extract lowercase keyword tokens from input text.

    This function normalizes text into alphanumeric tokens, removes common
    stopwords, and returns unique keywords used for lightweight lexical
    overlap checks during route selection.

    Args:
        text (str): Free-form text to tokenize and normalize.

    Returns:
        Set[str]: Unique keyword tokens with stopwords removed.
    """
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "have",
        "has",
        "had",
        "are",
        "was",
        "were",
        "you",
        "your",
        "about",
        "into",
        "than",
        "then",
        "they",
        "them",
        "their",
        "there",
        "would",
        "could",
        "should",
        "tell",
        "more",
        "please",
        "does",
        "did",
        "can",
        "not",
    }
    tokens = re.findall(r"[A-Za-z0-9]{3,}", text.lower())
    return {token for token in tokens if token not in stopwords}


def compute_keyword_overlap_ratio(question: str, documents: List[Document]) -> float:
    """Compute lexical overlap ratio between question and retrieved docs.

    The ratio is based on unique question keywords that also appear in the
    retrieved document content. Low overlap indicates likely topic mismatch,
    which should favor web-search fallback.

    Args:
        question (str): User question to evaluate.
        documents (List[Document]): Retrieved support documents.

    Returns:
        float: Overlap ratio in range [0.0, 1.0].
    """
    question_keywords = extract_keywords(question)
    if not question_keywords:
        return 1.0
    combined_context = " ".join([doc.page_content[:900] for doc in documents])
    context_keywords = extract_keywords(combined_context)
    overlap_count = len(question_keywords.intersection(context_keywords))
    return overlap_count / max(len(question_keywords), 1)


def rewrite_query_variants(llm: Optional[ChatGoogleGenerativeAI], question: str) -> List[str]:
    """Generate alternate query phrasings to improve retrieval recall.

    The original user question is always preserved, and up to two short
    alternate variants are generated to better match document phrasing.

    Args:
        llm (ChatGoogleGenerativeAI): LLM used for query rewriting.
        question (str): Original user question text.

    Returns:
        List[str]: Original question plus additional rewritten variants.
    """
    if llm is None:
        return [question]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite user questions for retrieval. "
                "Return up to 2 concise alternate queries, one per line, no numbering.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        raw_output = chain.invoke({"question": question}).strip()
    except Exception:
        return [question]

    variants = [question]
    for line in raw_output.splitlines():
        clean_line = line.strip("- ").strip()
        if clean_line and clean_line.lower() != question.lower() and clean_line not in variants:
            variants.append(clean_line)
        if len(variants) >= 3:
            break
    return variants


def retrieve_context(
    vector_store: PineconeVectorStore,
    llm: Optional[ChatGoogleGenerativeAI],
    question: str,
    namespace: str,
    min_relevance_threshold: float,
    min_keyword_overlap: float,
) -> RetrievedContext:
    """Retrieve relevant chunks and decide if web fallback is needed.

    This function runs multi-query similarity search, merges duplicate
    results, ranks by score, and computes a routing decision using a
    configurable relevance threshold.

    Args:
        vector_store (PineconeVectorStore): Vector store retrieval client.
        llm (ChatGoogleGenerativeAI): LLM used for query rewriting.
        question (str): User question to answer.
        namespace (str): Session namespace for scoped retrieval.
        min_relevance_threshold (float): Minimum confidence for doc-only answer.
        min_keyword_overlap (float): Minimum lexical overlap needed to trust
            retrieved context for in-document answering.

    Returns:
        RetrievedContext: Ranked documents, max score, and fallback flag.
    """
    merged_results = []
    for variant in rewrite_query_variants(llm, question):
        variant_results = vector_store.similarity_search_with_relevance_scores(
            query=variant,
            k=4,
            namespace=namespace,
        )
        merged_results.extend(variant_results)

    unique_best: Dict[tuple, tuple] = {}
    for doc, score in merged_results:
        key = (
            doc.metadata.get("doc_name"),
            doc.metadata.get("page_number"),
            doc.metadata.get("chunk_id"),
        )
        if key not in unique_best or score > unique_best[key][1]:
            unique_best[key] = (doc, score)

    ranked = sorted(unique_best.values(), key=lambda item: item[1], reverse=True)[:5]
    documents = [doc for doc, _ in ranked]
    max_score = ranked[0][1] if ranked else 0.0
    overlap_ratio = compute_keyword_overlap_ratio(question, documents)
    used_web_fallback = (
        (not documents)
        or (max_score < min_relevance_threshold)
        or (overlap_ratio < min_keyword_overlap)
    )

    return RetrievedContext(
        documents=documents,
        max_score=max_score,
        used_web_fallback=used_web_fallback,
    )


def format_document_citations(documents: List[Document]) -> List[str]:
    """Convert retrieved documents into readable citation labels.

    This function emits unique citations combining file name, inferred
    section title, and page number to satisfy human-friendly source display.

    Args:
        documents (List[Document]): Retrieved supporting documents.

    Returns:
        List[str]: Unique citation labels suitable for the UI.
    """
    seen = set()
    citations: List[str] = []
    for doc in documents:
        doc_name = doc.metadata.get("doc_name", "Unknown Document")
        page_number = doc.metadata.get("page_number", "?")
        section_title = doc.metadata.get("section_title", f"Page {page_number}")
        citation = f"{doc_name} — {section_title} (page {page_number})"
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)
    return citations

