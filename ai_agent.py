from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents import create_agent
from langchain.tools import tool

# LLM providers
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# RAG imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 🔥 Memory
memory = ConversationBufferMemory(return_messages=True)

# 🌐 Tavily instance
tavily = TavilySearch(max_results=5)

# 🌐 Web search tool
@tool
def tavily_search_tool(query: str):
    """Search the web for latest information"""
    results = tavily.invoke(query)
    return f"[SOURCE: WEB]\n{results}"


# 🚀 RAG Setup
VECTORSTORE = None  # Will be initialized when documents are loaded

def load_documents_rag(folder_path="docs"):
    """
    Load documents from a folder (PDF + TXT) and create FAISS vectorstore
    """
    global VECTORSTORE
    docs = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        print("No documents found in folder:", folder_path)
        return

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Build vectorstore
    VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    print(f"Loaded {len(chunks)} document chunks into VECTORSTORE")


# 🚀 MAIN FUNCTION
def get_response_from_ai_agent(
    llm_id,
    query,
    allow_search,
    system_prompts,
    provider,
    use_rag=True
):
    try:
        # ✅ Ensure query is string
        if isinstance(query, list):
            query = query[-1]

        # 🔥 Model selection
        if provider == "Groq":
            llm = ChatGroq(model=llm_id)
        elif provider == "hugging_face":
            hf_llm = HuggingFaceEndpoint(
                repo_id=llm_id,
                task="conversational",
                temperature=0.7
            )
            llm = ChatHuggingFace(llm=hf_llm)
        else:
            return "Invalid provider"

        # 🔥 Tools
        tools = []
        if allow_search:
            tools.append(tavily_search_tool)

        # 🔥 RAG retrieval
        context_text = ""
        if use_rag and VECTORSTORE:
            docs = VECTORSTORE.similarity_search(query, k=5)
            context_text = "\n".join([d.page_content for d in docs])

        # 🔥 Construct system prompt including retrieved context
        full_system_prompt = system_prompts
        if context_text:
            full_system_prompt += f"\n\n[Retrieved Context]\n{context_text}"

        # 🔥 Create agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=full_system_prompt
        )

        # ✅ Construct message state
        state = {
            "messages": memory.chat_memory.messages + [
                HumanMessage(content=query)
            ]
        }

        # 🔥 Invoke agent
        response = agent.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
        ai_response = ai_messages[-1] if ai_messages else "No response"

        # 🔥 Source detection
        if "[SOURCE: WEB]" in ai_response:
            ai_response = ai_response.replace("[SOURCE: WEB]", "")
            ai_response += "\n\nSources:\n- Web Search"
        elif context_text:
            ai_response += "\n\nSources:\n- RAG Documents"
        else:
            ai_response += "\n\nSources:\n- General Knowledge"

        # 🔥 Save memory
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(ai_response)

        return ai_response

    except Exception as e:
        print("Agent error:", e)
        return f"Error: {str(e)}"