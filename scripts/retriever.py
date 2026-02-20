import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
script_dir = Path(__file__).parent
CHROMA_DB_DIR = Path(f"{script_dir.parent}/chroma_langchain_db")
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"
COLLECTION_NAME = "earnings_collection"

load_dotenv()

# Initialize Global Clients
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

@tool(response_format="content_and_artifact")
def retrieve_earnings_context(query: str, ticker: str = None, quarter: str = None, year: int = None):
    """
    Retrieve information to help answer a query about EV earnings calls.
    Optionally filter by ticker (e.g., TSLA, RIVN), quarter (e.g., Q4), and year.
    """
    # 1. Build individual conditions
    conditions = []
    if ticker:
        conditions.append({"ticker": ticker.upper()})
    if quarter:
        conditions.append({"quarter": quarter.upper()})
    if year:
        conditions.append({"year": year})

    # 2. Format the filter dict according to ChromaDB syntax
    filter_dict = {}
    if len(conditions) == 1:
        # Single condition can be passed directly
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        # Multiple conditions must be wrapped in an $and operator
        filter_dict = {"$and": conditions}

    # 3. Configure the retriever with the dynamic filter
    search_kwargs = {"k": 3}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
        logger.info(f"Executing search with filter: {filter_dict}")
    else:
        logger.info("Executing search across all documents (no filter applied).")

    # 4. Execute the filtered search
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(query)
    
    # 5. Serialize for the LLM
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def get_rag_agent():
    """Initializes and returns the Langchain agent executor."""
    # Set temperature to 0 for factual financial retrieval
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    tools = [retrieve_earnings_context]
    
    # Use a ChatPromptTemplate to give the agent strict operational boundaries
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst assistant. You have access to a tool that retrieves context from company earnings calls transcripts. "
                   "Always use the `retrieve_earnings_context` tool to find factual answers. If the user specifies a company, quarter, or year, pass those as arguments to the tool."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Modern tool calling agent formulation
    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

if __name__ == "__main__":
    # Test the agent locally
    agent_executor = get_rag_agent()
    
    test_query = "What did Robert Scaringe say about the R2 in Q4 2025?"
    logger.info(f"Testing Query: {test_query}")
    
    response = agent_executor.invoke({"input": test_query})
    print("\nFinal Answer:\n", response["output"])