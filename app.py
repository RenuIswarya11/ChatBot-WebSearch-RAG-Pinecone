from typing import Any, Tuple

import streamlit as st


def ensure_runtime_clients(config: Any) -> Tuple[Any, Any]:
    """Lazily initialize vector store and LLM clients.

    This helper delays heavy client/model initialization until the user
    actually needs indexing or question-answering. It stores initialized
    clients in Streamlit session state so future actions reuse them without
    repeating startup cost.

    Args:
        config (Any): Loaded application configuration object with API keys
            and provider settings.

    Returns:
        Tuple[Any, Any]: Initialized `vector_store` and `llm` client objects.
    """
    if "vector_store" in st.session_state and "llm" in st.session_state:
        return st.session_state.vector_store, st.session_state.llm

    from src.services.vector_store_service import build_store_and_models

    vector_store, llm = build_store_and_models(config)
    st.session_state.vector_store = vector_store
    st.session_state.llm = llm
    return vector_store, llm


def main() -> None:
    """Run the Streamlit application and orchestrate all workflows.

    This entrypoint loads configuration, initializes clients and session
    memory, handles document ingestion, and serves RAG chat with fallback.

    Args:
        None: Runtime dependencies are initialized inside the function.

    Returns:
        None: UI is rendered and actions are handled via Streamlit events.
    """
    from src.config import load_app_config
    from src.state import initialize_session_state
    from src.ui import configure_page, render_chat_history, render_header, render_sidebar

    configure_page()
    initialize_session_state()
    render_header()

    try:
        config = load_app_config()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    clear_requested = render_sidebar(
        config=config,
        indexed=st.session_state.indexed,
        indexed_documents=st.session_state.indexed_documents,
    )
    if clear_requested:
        st.session_state.messages = []
        st.rerun()

    with st.expander("1) Upload Documents", expanded=True):
        uploads = st.file_uploader(
            f"Upload up to {config.max_files} PDF files (max {config.max_pages_per_file} pages each)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            from src.services.document_service import validate_uploads

            valid, message = validate_uploads(uploads, config)
            if not valid:
                st.error(message)
            else:
                st.success(message)
                if st.button("Index documents", type="primary", use_container_width=True):
                    with st.spinner("Extracting, chunking, embedding, and storing in Pinecone..."):
                        try:
                            from src.services.document_service import (
                                chunk_documents,
                                parse_pdfs_to_documents,
                                upsert_chunks,
                            )

                            vector_store, _llm = ensure_runtime_clients(config)
                            documents = parse_pdfs_to_documents(uploads)
                            if not documents:
                                st.error("No readable text was found in the uploaded PDFs.")
                                return
                            chunks = chunk_documents(documents, config)
                            inserted = upsert_chunks(
                                vector_store=vector_store,
                                chunks=chunks,
                                namespace=st.session_state.session_id,
                            )
                        except Exception as exc:
                            st.error(f"Indexing failed: {exc}")
                            return
                    st.session_state.indexed = inserted > 0
                    st.session_state.indexed_documents = [file.name for file in uploads]
                    st.success(f"Indexed {inserted} chunks successfully.")

    st.divider()
    st.subheader("2) Ask Questions")
    if not st.session_state.indexed:
        st.info("Upload and index at least one PDF to start Q&A.")

    render_chat_history(st.session_state.messages)

    user_question = st.chat_input("Ask a question about your uploaded documents...")
    if user_question and not st.session_state.indexed:
        st.warning("Please upload and index at least one PDF before asking questions.")
        return

    if user_question and st.session_state.indexed:
        from src.services.qa_service import build_answer_payload
        from src.services.retrieval_service import retrieve_context

        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning over your documents..."):
                vector_store, llm = ensure_runtime_clients(config)
                retrieved_context = retrieve_context(
                    vector_store=vector_store,
                    llm=llm,
                    question=user_question,
                    namespace=st.session_state.session_id,
                    min_relevance_threshold=config.min_relevance_threshold,
                    min_keyword_overlap=config.min_keyword_overlap,
                )
                payload = build_answer_payload(
                    llm=llm,
                    question=user_question,
                    chat_history=st.session_state.messages,
                    retrieved_context=retrieved_context,
                    serpapi_api_key=config.serpapi_api_key,
                )

            st.markdown(payload.text)
            st.markdown("**Sources**")
            for citation in payload.citations:
                st.markdown(f"- {citation}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": payload.text,
                    "citations": payload.citations,
                }
            )


if __name__ == "__main__":
    main()
