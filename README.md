# Smart Research Assistant (RAG-based Q&A System)

This project is a Streamlit-based application that allows users to upload PDF documents and ask questions based on them. The system uses a Retrieval-Augmented Generation (RAG) approach to provide answers grounded in the uploaded documents, and falls back to web search when needed.

---

## Features

- Upload up to 5 PDF documents (with page limit validation)
- Ask questions based on document content
- Answers include proper source citations
- Automatic fallback to web search if documents are not sufficient
- Maintains conversation history within the session

---

## Tech Stack

- LangChain (for orchestration)
- Gemini (LLM + embeddings)
- Pinecone (vector database)
- Streamlit (frontend)
- SerpAPI (web search fallback)

---

## Chunking Strategy

I used a chunk size of **900 characters** with an overlap of **180 characters** using a recursive text splitter.

The reason for choosing this:
- Smaller chunks improve retrieval accuracy
- Overlap ensures that important context is not lost between chunks
- Works well for PDFs where content may be split across pages

---

## Retrieval Approach

I used **similarity search in Pinecone** to retrieve relevant chunks.

Additionally:
- I implemented **query rewriting** to generate alternative versions of the user query
- Then I merged results from multiple queries to improve retrieval

This helps because:
- Users may phrase questions differently than document content
- Multi-query retrieval improves recall without adding too much complexity

---

## Agent Routing Logic

The system decides whether to use document-based answers or web search based on:

1. If no relevant chunks are retrieved  
2. If the relevance score is below a threshold  
3. If keyword overlap between query and retrieved text is low  

If any of these conditions are met:
- The system switches to **web search (SerpAPI)**
- It also informs the user that documents were insufficient

---

## Problems Faced and Solutions

**1. Poor citation clarity**
- Initially, answers showed raw chunks which were hard to understand
- I fixed this by attaching metadata like document name and page number

**2. Irrelevant retrieval results**
- Sometimes similarity search returned weak matches
- I introduced a **minimum relevance threshold** and keyword overlap check

**3. Query mismatch issue**
- User queries didn’t always match document wording
- Solved using **query rewriting + multi-query retrieval**

---

## One Improvement With More Time

If I had more time, I would add a **re-ranking layer (cross-encoder)** after retrieval.

This would:
- Improve answer accuracy
- Reduce irrelevant chunks in final context

---

## Test Documents

The `test_documents/` folder contains sample PDFs used for testing:

- `rag_design_patterns.pdf`
- `agent_routing_strategies.pdf`
- `evaluation_groundedness.pdf`

These were used to verify:
- Retrieval quality
- Citation accuracy
- Fallback behavior

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
