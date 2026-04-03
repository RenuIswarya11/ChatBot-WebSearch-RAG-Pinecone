# Smart Research Assistant (LangChain + Gemini + Pinecone + SerpAPI)

RAG-powered Streamlit application for document-grounded Q&A with citation-first answers and web fallback when uploaded documents are insufficient.

## Live URL

Add your deployed URL after publishing (Streamlit Cloud recommended).

## Requirement Compliance Checklist

- Document Upload: Implemented (`max 5 PDFs`, `max 10 pages per file` validation before indexing).
- Chunking and Embedding: Implemented (recursive chunking + Gemini embeddings + Pinecone upsert).
- RAG Q&A: Implemented (retrieval from uploaded documents only, session namespace scoped).
- Source Citations: Implemented for every answer (document citations and/or web citations).
- Agent Fallback with Web Search: Implemented (threshold-based route to SerpAPI with explicit message).
- Conversation Memory: Implemented (session chat history passed into LLM for follow-up queries).
- Polished UI: Implemented (structured layout, spacing, status cards, loading/empty states, responsive width).
- Bonus Feature: Implemented (query rewriting + multi-query retrieval fusion).

## Tech Stack

- Framework: `LangChain`
- LLM: `Gemini` (`gemini-2.5-flash`)
- Embeddings: `models/gemini-embedding-001` (with fallback to `sentence-transformers/all-MiniLM-L6-v2`)
- Vector Store: `Pinecone` (free tier compatible)
- Frontend: `Streamlit`
- Agent Web Tool: `SerpAPI`

## Why LangChain

LangChain provides fast integration for prompt chains, model wrappers, vector-store retrieval, and orchestration needed for RAG + fallback behavior in a single Python codebase.

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── state.py
│   ├── ui.py
│   └── services/
│       ├── __init__.py
│       ├── document_service.py
│       ├── vector_store_service.py
│       ├── retrieval_service.py
│       ├── web_search_service.py
│       └── qa_service.py
└── test_documents/
    ├── research_methods.pdf
    ├── rag_design_patterns.pdf
    ├── vector_search_fundamentals.pdf
    ├── agent_routing_strategies.pdf
    ├── evaluation_groundedness.pdf
    ├── deployment_playbook.pdf
    └── mlops_observability.pdf
```

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables (`.env`)

```env
GEMINI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here
PINECONE_INDEX_NAME=smart-research-assistant
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.5-flash
CHUNK_SIZE=900
CHUNK_OVERLAP=180
MIN_RELEVANCE_THRESHOLD=0.30
MIN_KEYWORD_OVERLAP=0.12
MAX_FILES=5
MAX_PAGES_PER_FILE=10
```

### 3) Run locally

```bash
streamlit run app.py
```

## Chunking Strategy

- Chunk size: `900`
- Overlap: `180`
- Splitter: `RecursiveCharacterTextSplitter`
- Rationale: preserves enough context for answer quality while maintaining focused semantic chunks for accurate retrieval; overlap protects boundary information.

## Retrieval Approach

- Similarity search in Pinecone with relevance scores.
- Query rewriting creates up to two alternates for each user question.
- Result fusion deduplicates by source chunk and keeps top-ranked evidence.

Why this approach:
- Better recall on user phrasing mismatch.
- Stable latency and straightforward production behavior.
- Works well with free-tier constraints.

## Agent Routing Logic (Documents vs Web)

1. Retrieve top chunks from session namespace.
2. Compute best relevance score.
3. Compute lexical overlap between question keywords and retrieved context.
4. If no chunk found, score `< MIN_RELEVANCE_THRESHOLD`, or overlap `< MIN_KEYWORD_OVERLAP`, route to web search.
4. Explicitly tell user docs were insufficient.
5. Generate answer from available web snippets.
6. Always show sources.

## Source Citation Strategy

Each indexed chunk keeps:
- `doc_name`
- `page_number`
- `section_title` (best-effort inferred from early page lines)

Citation format:
- Document: `<doc_name> — <section_title> (page <n>)`
- Web fallback: `Web: <title> (<url>)`

## Conversation Memory

Memory is implemented using Streamlit `session_state`, which is enough for this assignment because the requirement is **within-session** continuity.

Redis is optional, not mandatory:
- Use Redis if you need memory persistence across browser refreshes, multi-instance scaling, or long-lived server sessions.
- For this task, session memory is correct and simpler.

## Bonus Feature

Query rewriting + multi-query retrieval fusion:
- The app rewrites questions and merges retrieval results across variants.
- This improves recall and makes answers more robust to wording differences.

## Problems Faced and Solutions

1. Citation readability:
   - Problem: raw chunks are hard to interpret.
   - Solution: attach and render document name + section + page metadata.

2. Weak document evidence:
   - Problem: forcing a doc-only answer risks hallucinations.
   - Solution: confidence threshold + explicit web fallback route.

3. Query phrasing mismatch:
   - Problem: user wording often differs from document language.
   - Solution: query rewriting before retrieval.

## One Improvement With More Time

Add reranking (for example cross-encoder rerank stage) before final context selection to increase precision for dense and long documents.

## Assessment README Coverage (Requested)

This section maps exactly to the assessment checklist:

- **Your chunking strategy — what chunk size, what overlap, and why**
  - Covered in `Chunking Strategy` section.
  - Values: `CHUNK_SIZE=900`, `CHUNK_OVERLAP=180`, plus rationale.

- **Your retrieval approach — similarity search, re-ranking, or something else? Why?**
  - Covered in `Retrieval Approach` section.
  - Method: Pinecone similarity search + multi-query rewriting + result fusion.

- **Agent routing logic — how does the agent decide when to use web search vs. document context?**
  - Covered in `Agent Routing Logic (Documents vs Web)` section.
  - Rule combines relevance threshold and keyword-overlap threshold.

- **Problems you ran into and how you solved them**
  - Covered in `Problems Faced and Solutions` section.

- **One thing you’d improve if you had more time**
  - Covered in `One Improvement With More Time` section.

- **A test_documents/ folder in the repo with 2–3 sample PDFs you used for testing**
  - `test_documents/` is present with multiple topic-specific PDFs.
  - For reviewer quick check, these three are recommended:
    - `rag_design_patterns.pdf`
    - `agent_routing_strategies.pdf`
    - `evaluation_groundedness.pdf`

- **A short Loom video (under 5 minutes) walking through your app and code**
  - Add your Loom link below before submission:
  - `Loom URL: <paste-your-loom-link>`

## Deployment (What to Submit)

Deploy on Streamlit Cloud:
- Push repo to GitHub (public).
- Create Streamlit app with entrypoint `app.py`.
- Add environment variables in Streamlit secrets.
- Share deployed URL.

Submission checklist:
- Live deployed URL
- Public GitHub repo
- Completed `README.md` (this file)
- `test_documents/` with sample PDFs (at least 2-3; this repo includes more for broader testing)
- Loom walkthrough video under 5 minutes (`Loom URL` field above)

