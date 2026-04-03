import uuid

import streamlit as st


def initialize_session_state() -> None:
    """Initialize all required Streamlit session state keys.

    This function ensures the app has stable keys for session namespace,
    indexed state, chat messages, and indexed document names before any UI
    or retrieval operation is executed.

    Args:
        None: The function writes default values into `st.session_state`.

    Returns:
        None: Session state is updated in-place.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []

