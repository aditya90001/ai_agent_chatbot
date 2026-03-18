from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from ai_agent import get_response_from_ai_agent

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
    "mixtral-8x7b-32768",
    "deepseek-ai/DeepSeek-R1"
]

# ✅ Request schema
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


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
            query=request.messages[-1],   # ✅ FIX
            allow_search=request.allow_search,
            system_prompts=request.system_prompt,
            provider=request.model_provider,
        )

        # ✅ ALWAYS RETURN JSON
        return {"response": response}

    except Exception as e:
        print("Backend error:", e)
        return {"error": str(e)}