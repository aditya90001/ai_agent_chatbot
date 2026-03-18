import streamlit as st
import requests
import time

# 🔥 Config
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 ChatGPT-like AI")

API_URL = "http://127.0.0.1:9999/chat"

# 🔥 Sidebar (Settings)
st.sidebar.header("⚙️ Settings")

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful AI assistant"
)

provider = st.sidebar.selectbox("Provider", ["Groq", "hugging_face"])

if provider == "Groq":
    model = st.sidebar.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )
else:
    model = st.sidebar.selectbox(
        "Model",
        ["deepseek-ai/DeepSeek-R1"]
    )

allow_search = st.sidebar.checkbox("Enable Web Search")

# 🔥 Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# 🔥 Chat Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔥 Display previous messages
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# 🔥 Chat input
user_input = st.chat_input("Type your message...")

if user_input:

    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # 🔥 Prepare request
    payload = {
        "model_name": model,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": [user_input],
        "allow_search": allow_search
    }

    # 🔥 Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = requests.post(API_URL, json=payload)
                data = response.json()

                if "error" in data:
                    reply = data["error"]
                else:
                    reply = data["response"]

                # 🔥 Separate sources
                if "Sources:" in reply:
                    main, sources = reply.split("Sources:", 1)
                else:
                    main, sources = reply, None

                # 🔥 Typing effect
                placeholder = st.empty()
                typed_text = ""

                for char in main:
                    typed_text += char
                    placeholder.markdown(typed_text)
                    time.sleep(0.005)

                # 🔥 Show sources in expander
                if sources:
                    with st.expander("📚 Sources"):
                        st.markdown(sources)

            except Exception as e:
                reply = f"Connection error: {str(e)}"
                st.error(reply)

    # 🔥 Save response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply}
    )