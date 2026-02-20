import json
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Configure logging for production visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
script_dir = Path(__file__).parent
TRANSCRIPTS_DIR = Path(f"{script_dir.parent}/transcripts")
CHROMA_DB_DIR = Path(f"{script_dir.parent}/chroma_langchain_db")
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "earnings_collection"

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extracts ticker, quarter, and year from a filename like 'RIVN_Q4_2025.json'.
    """
    # Regex to capture TICKER_Q#_YYYY
    match = re.match(r"([A-Z]+)_(Q[1-4])_(\d{4})", filename)
    if match:
        return {
            "ticker": match.group(1),
            "quarter": match.group(2),
            "year": int(match.group(3))
        }
    
    logger.warning(f"Could not extract standard metadata from filename: {filename}")
    return {}

def load_and_process_documents(data_dir: Path) -> list[Document]:
    """
    Loads JSON transcripts, extracts metadata, and returns Langchain Document objects.
    """
    documents = []
    
    for file_path in data_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            text_content = data.get("full_conference_call_transcript", "")
            if not text_content:
                logger.warning(f"No 'full_conference_call_transcript' key found in {file_path.name}")
                continue

            # 1. Extract dynamic metadata from the file name
            metadata = extract_metadata_from_filename(file_path.name)
            metadata["source"] = str(file_path.name)

            # 2. Bind the metadata to the full document before splitting
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
            logger.info(f"Loaded {file_path.name} with metadata: {metadata}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            
    return documents

def ingest_data():
    """Main pipeline to load, chunk, and embed documents."""
    load_dotenv()
    logger.info("Starting RAG ingestion pipeline...")
    
    # 1. Load Documents & Attach Metadata
    raw_documents = load_and_process_documents(TRANSCRIPTS_DIR)
    if not raw_documents:
        logger.error("No documents found to process. Exiting.")
        return

    # 2. Split Documents
    # Because metadata is attached to the parent Document, Langchain will 
    # automatically cascade this metadata down to every single chunk.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunked_docs = text_splitter.split_documents(raw_documents)
    logger.info(f"Successfully split {len(raw_documents)} transcripts into {len(chunked_docs)} chunks.")

    # 3. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Ingest into Vector Store
    logger.info(f"Ingesting embeddings into ChromaDB at {CHROMA_DB_DIR}...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    
    vector_store.add_documents(documents=chunked_docs)
    logger.info("Ingestion complete. Vector store is hydrated and ready for queries.")

if __name__ == "__main__":
    ingest_data()