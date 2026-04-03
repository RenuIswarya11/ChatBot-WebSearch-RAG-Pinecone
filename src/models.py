from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document


@dataclass
class RetrievedContext:
    """Store retrieval output used for answer routing."""

    documents: List[Document]
    max_score: float
    used_web_fallback: bool


@dataclass
class AnswerPayload:
    """Represent the final answer and its citations."""

    text: str
    citations: List[str]
    diagnostics: Dict[str, str]

