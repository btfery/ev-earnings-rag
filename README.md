# EV Earnings RAG API

## Overview
The EV Earnings RAG API is a Retrieval-Augmented Generation (RAG) application designed to answer questions specifically about Electric Vehicle (EV) earnings call transcripts. 

## Features
* **Smart Ingestion Pipeline**: Automatically extracts `ticker`, `quarter`, and `year` metadata from transcript filenames (e.g., `TSLA_Q4_2025.json`) and attaches it to document chunks.
* **Dynamic Metadata Filtering**: The LLM automatically extracts entities from the user's prompt (like the company name or year) and applies ChromaDB `$and` filters to narrow the search space before performing vector similarity search.
* **FastAPI Backend**: Provides a robust, typed API endpoint (`/chat`) for seamless integration with frontends or other microservices.
* **Google Gemini Integration**: Utilizes `gemini-2.5-flash-lite` for agentic reasoning and `gemini-embedding-001` for semantic search.

## Tech Stack
* **Framework:** [LangChain](https://www.langchain.com/)
* **LLM & Embeddings:** [Google Generative AI (Gemini)](https://ai.google.dev/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **API Routing:** [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)

## Project Structure
```text
.
├── scripts/
│   ├── ingest.py           # Data loading, metadata extraction, chunking, and embedding
│   └── retriever.py        # LangChain tool-calling agent and ChromaDB retrieval logic
├── transcripts/            # Directory containing raw JSON earnings transcripts
├── chroma_langchain_db/    # Locally persisted Chroma vector database (auto-generated)
└── app.py                  # FastAPI application and endpoint definitions