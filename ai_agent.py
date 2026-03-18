from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents import create_agent
from langchain.tools import tool

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


# 🚀 MAIN FUNCTION
def get_response_from_ai_agent(
    llm_id,
    query,
    allow_search,
    system_prompts,
    provider,
    vectorstore=None
):

    try:
        # ✅ FIX 1: Ensure query is string
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

        # 🔥 Agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompts
        )

        # ✅ FIX 2: Use proper message object
        state = {
            "messages": memory.chat_memory.messages + [
                HumanMessage(content=query)
            ]
        }

        # 🔥 Invoke agent
        response = agent.invoke(state)

        messages = response.get("messages", [])

        ai_messages = [
            m.content for m in messages if isinstance(m, AIMessage)
        ]

        ai_response = ai_messages[-1] if ai_messages else "No response"

        # 🔥 Source detection
        if "[SOURCE: WEB]" in ai_response:
            ai_response = ai_response.replace("[SOURCE: WEB]", "")
            ai_response += "\n\nSources:\n- Web Search"
        else:
            ai_response += "\n\nSources:\n- General Knowledge"

        # 🔥 Save memory properly
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(ai_response)

        return ai_response

    except Exception as e:
        print("Agent error:", e)
        return f"Error: {str(e)}"