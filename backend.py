from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from ai_agent import get_response_from_ai_agent, load_documents_rag

app = FastAPI(title="Agentic AI Chatbot")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Allowed models
ALLOWED_MODEL_NAMES = [
    "llama-3.3-70b-versatile",
    
    "deepseek-ai/DeepSeek-R1"
]

# ✅ Request schema
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool
    use_rag: bool = True  # <-- NEW PARAMETER

# 🔥 Load RAG documents on server start
@app.on_event("startup")
def startup_event():
    print("Loading RAG...")
    load_documents_rag(folder_path="docs")
    print("RAG Loaded ✅")  # Make sure 'docs/' folder exists

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat_endpoint(request: RequestState):

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name"}

    try:
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=request.messages[-1],
            allow_search=request.allow_search,
            system_prompts=request.system_prompt,
            provider=request.model_provider,
            use_rag=request.use_rag
        )

        return {"response": response}

    except Exception as e:
        print("Backend error:", e)
        return {"error": str(e)}