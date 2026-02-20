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
```
----

## Getting Started

### 1. Create a Virtual Environment
Open your terminal at the root of the project and run:
```
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required packages for the LangChain pipeline, ChromaDB, and the FastAPI server:

```
uv lock
```

### 3. Set Up Environment Variables
Create a file named .env in the root directory of the project. Add your Google Gemini API key to this file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Ingest the Data
Run the ingestion script. This will load the JSON files, split the text into chunks, attach the extracted metadata, generate embeddings, and save the local Chroma vector database to ./chroma_langchain_db.
```
python scripts/ingest.py
```

### 5. Start the Server
Once the data is ingested, launch the FastAPI server using Uvicorn:
```
python app.py
```
The server will start locally and listen at http://0.0.0.0:8000.

----

## API Usage & Testing
With the server running, you can interact with the RAG agent via the `/chat` endpoint. You can also view the interactive Swagger UI documentation by navigating to `http://0.0.0.0:8000/docs` in your web browser.

### Option 1: Testing via cURL (Terminal)
Open a new terminal window (leave the server running in the original window) and execute:
```
curl -X POST "[http://0.0.0.0:8000/chat](http://0.0.0.0:8000/chat)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What did Robert Scaringe say about the R2 in Q4 2025?"}'
```

### Option 2: Testing via Python
You can test the endpoint programmatically using a separate Python script:
```
import requests

url = "[http://0.0.0.0:8000/chat](http://0.0.0.0:8000/chat)"
payload = {"query": "What did Robert Scaringe say about the R2 in Q4 2025?"}

response = requests.post(url, json=payload)
if response.status_code == 200:
    print(response.json().get("answer"))
else:
    print(f"Error: {response.text}")
```

### Server Management
Shutting Down: To stop the FastAPI server gracefully, return to the terminal running the server and press `Ctrl + C`.
