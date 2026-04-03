from typing import Dict, List

import streamlit as st

from src.config import AppConfig


def configure_page() -> None:
    """Configure Streamlit page metadata and polished CSS styles.

    This function sets global page properties and injects lightweight CSS
    for spacing, typography, cards, pills, and responsive content width.

    Args:
        None: UI configuration is applied globally to current page.

    Returns:
        None: Streamlit page settings and styles are applied in-place.
    """
    st.set_page_config(page_title="Smart Research Assistant", page_icon=":books:", layout="wide")
    st.markdown(
        """
        <style>
        .main { max-width: 1100px; margin: 0 auto; }
        .hero-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem; }
        .hero-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1rem; }
        .chip {
            display: inline-block; border-radius: 999px; padding: 0.2rem 0.6rem;
            font-size: 0.8rem; margin-right: 0.4rem; margin-bottom: 0.4rem;
            background: #eef2ff; color: #3730a3;
        }
        .panel {
            border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.9rem 0.95rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render application title and descriptive subtitle.

    This function displays the hero section introducing capabilities and
    fallback behavior, creating a clear first impression for evaluators.

    Args:
        None: Header uses static app messaging.

    Returns:
        None: Header elements are rendered into the page.
    """
    st.markdown("<div class='hero-title'>Smart Research Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Upload PDFs, ask natural questions, and get cited answers. "
        "When document evidence is weak, the assistant uses web search fallback.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(config: AppConfig, indexed: bool, indexed_documents: List[str]) -> bool:
    """Render sidebar controls and return clear-chat action state.

    The sidebar summarizes runtime configuration, session namespace, and
    current indexing status to make debugging and demos easier.

    Args:
        config (AppConfig): Runtime configuration values.
        indexed (bool): Whether current session has indexed content.
        indexed_documents (List[str]): Indexed file names for session.

    Returns:
        bool: True when user requests chat clear; otherwise False.
    """
    with st.sidebar:
        st.header("Workspace")
        st.caption("LangChain + Gemini + Pinecone + SerpAPI")
        st.markdown(f"`index`: {config.pinecone_index_name}")
        st.markdown(f"`threshold`: {config.min_relevance_threshold}")
        st.markdown(
            "<div class='panel'>"
            f"<b>Status:</b> {'Indexed' if indexed else 'Not Indexed'}<br/>"
            f"<b>Docs:</b> {len(indexed_documents)}"
            "</div>",
            unsafe_allow_html=True,
        )
        for name in indexed_documents:
            st.markdown(f"<span class='chip'>{name}</span>", unsafe_allow_html=True)
        return st.button("Clear chat history")


def render_chat_history(messages: List[Dict[str, str]]) -> None:
    """Render historical chat messages with citations if available.

    This function replays assistant and user turns from session state so
    follow-up interactions preserve complete conversational context in UI.

    Args:
        messages (List[Dict[str, str]]): Serialized chat turns from state.

    Returns:
        None: Chat messages are rendered in chronological order.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Sources**")
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

