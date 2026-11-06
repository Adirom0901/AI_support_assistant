
import logging
import os
import secrets
import threading
import uuid
 
from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
 
from graph import graph          # importing this loads the model (takes a while on startup)
from model import MODEL_PATH
 
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)
 
API_KEY = os.getenv("API_KEY", "")   # empty = no authentication
_model_lock = threading.Lock()       # one generation at a time (model/GPU is not thread-safe)
 
app = FastAPI(title="Support Assistant API", version="1.0.0")
 
 
def check_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
 
 
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, description="Reuse to continue a conversation")
 
 
class ChatResponse(BaseModel):
    reply: str
    thread_id: str
 
 
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}
 
 
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_api_key)])
def chat(req: ChatRequest):
    # Plain `def` (not async): FastAPI runs it in a worker thread, so the server stays responsive.
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        with _model_lock:
            result = graph.invoke(
                {"messages": [HumanMessage(content=req.message)]}, config=config
            )
    except Exception:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail="Assistant failed to generate a reply")
 
    return ChatResponse(reply=result["messages"][-1].content, thread_id=thread_id)