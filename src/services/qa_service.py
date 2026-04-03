from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.models import AnswerPayload, RetrievedContext
from src.services.retrieval_service import format_document_citations
from src.services.web_search_service import search_web_with_serpapi


def compose_answer(
    llm: Optional[ChatGoogleGenerativeAI],
    question: str,
    chat_history: List[Dict[str, str]],
    context_documents: List[Document],
    web_results: List[Dict[str, str]],
) -> str:
    """Generate a grounded response from document and web context.

    The prompt includes bounded recent conversation turns for follow-up
    resolution, plus retrieved context. It instructs the model to be
    explicit when confidence is low.

    Args:
        llm (ChatGoogleGenerativeAI): LLM instance used for generation.
        question (str): Current user question.
        chat_history (List[Dict[str, str]]): Prior chat turns in session.
        context_documents (List[Document]): Retrieved document snippets.
        web_results (List[Dict[str, str]]): Optional web snippets for fallback.

    Returns:
        str: Model-generated assistant response in markdown text.
    """
    history_text = "\n".join(
        [f"{message['role'].upper()}: {message['content']}" for message in chat_history[-8:]]
    )
    doc_context = "\n\n".join([doc.page_content[:1300] for doc in context_documents])
    web_context = "\n\n".join(
        [f"Title: {item['title']}\nURL: {item['link']}\nSnippet: {item['snippet']}" for item in web_results]
    )

    if llm is None:
        if context_documents:
            snippets = "\n\n".join([doc.page_content[:400] for doc in context_documents[:2]])
            return (
                "LLM generation is currently unavailable for this API key. "
                "Showing the most relevant extracted document context:\n\n"
                + snippets
            )
        if web_results:
            first = web_results[0]
            return (
                "LLM generation is currently unavailable for this API key. "
                "Showing top web fallback result:\n\n"
                f"{first.get('title', 'Untitled')}\n{first.get('snippet', '')}\n{first.get('link', '')}"
            )
        return "LLM generation is unavailable and no context could be retrieved."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise research assistant. "
                "Answer only from provided context and state uncertainty clearly when context is weak.",
            ),
            (
                "human",
                "Conversation history:\n{history}\n\n"
                "Question:\n{question}\n\n"
                "Document context:\n{doc_context}\n\n"
                "Web context:\n{web_context}\n\n"
                "Write a concise, useful answer.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke(
            {
                "history": history_text or "No previous conversation.",
                "question": question,
                "doc_context": doc_context or "No reliable document context found.",
                "web_context": web_context or "No web results available.",
            }
        )
    except Exception:
        if context_documents:
            excerpt = context_documents[0].page_content[:500]
            return (
                "I could not call the configured language model for generation right now. "
                "Here is a direct excerpt from the most relevant uploaded document:\n\n"
                f"{excerpt}"
            )
        if web_results:
            first = web_results[0]
            return (
                "I could not call the configured language model for generation right now. "
                "Here is the top web result summary:\n\n"
                f"{first.get('title', 'Untitled')}\n{first.get('snippet', '')}\n{first.get('link', '')}"
            )
        return (
            "I could not call the configured language model for generation right now, "
            "and no usable context was available."
        )


def build_answer_payload(
    llm: Optional[ChatGoogleGenerativeAI],
    question: str,
    chat_history: List[Dict[str, str]],
    retrieved_context: RetrievedContext,
    serpapi_api_key: str,
) -> AnswerPayload:
    """Build final answer payload with fallback handling and citations.

    This function handles routing behavior between document-grounded answers
    and web fallback answers, then guarantees that the response includes
    source citations in all scenarios.

    Args:
        llm (ChatGoogleGenerativeAI): LLM used for answer generation.
        question (str): Current user question.
        chat_history (List[Dict[str, str]]): Session conversation history.
        retrieved_context (RetrievedContext): Retrieval output and routing flag.
        serpapi_api_key (str): API key for web fallback calls.

    Returns:
        AnswerPayload: Final answer text, citation list, and diagnostics.
    """
    web_results: List[Dict[str, str]] = []
    routing_note = "Used uploaded document context."
    web_error = ""

    if retrieved_context.used_web_fallback:
        routing_note = (
            "Document evidence was weak, so web search fallback was attempted."
        )
        try:
            web_results = search_web_with_serpapi(serpapi_api_key, question, top_k=5)
        except Exception as exc:
            web_error = str(exc)

    answer_text = compose_answer(
        llm=llm,
        question=question,
        chat_history=chat_history,
        context_documents=retrieved_context.documents,
        web_results=web_results,
    )

    if retrieved_context.used_web_fallback:
        answer_text = (
            "I could not find strong support in your uploaded documents. "
            "I attempted web search and answered from available web context.\n\n"
            + answer_text
        )

    citations = format_document_citations(retrieved_context.documents)
    if web_results:
        citations.extend([f"Web: {item['title']} ({item['link']})" for item in web_results[:3]])
    if not citations:
        citations = ["Source unavailable: web search attempted but returned no reliable references."]

    return AnswerPayload(
        text=answer_text,
        citations=citations,
        diagnostics={
            "route": "web_fallback" if retrieved_context.used_web_fallback else "documents",
            "max_score": f"{retrieved_context.max_score:.3f}",
            "routing_note": routing_note,
            "web_error": web_error,
        },
    )

