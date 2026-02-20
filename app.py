import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import our previously built agent executor
from scripts.retriever import get_rag_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EV Earnings RAG API",
    description="An AI agent that answers questions about EV earnings calls.",
    version="1.0.0"
)

# 1. Define the data contract using Pydantic
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Initialize the agent once at startup
try:
    agent_executor = get_rag_agent()
    logger.info("RAG Agent successfully initialized.")
except Exception as e:
    logger.error(f"Failed to initialize RAG Agent: {e}")
    agent_executor = None

# 2. Define the API Endpoint
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Accepts a user query, passes it to the Langchain agent, and returns the response.
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent is not initialized.")
    
    logger.info(f"Received query: {request.query}")
    
    try:
        # 3. Invoke the Langchain agent
        response = agent_executor.invoke({"input": request.query})
        
        # The agent's final answer is typically stored in the 'output' key
        answer = response.get("output", "I could not generate an answer.")
        return QueryResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API locally using Uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)