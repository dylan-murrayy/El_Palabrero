import datetime

import streamlit as st
from openai import OpenAI

from chat import chat_handling
from storage import initialize_database, load_chat, load_saved_chats, save_chat
from ui_components import (
    analytics_dashboard,
    display_chat,
    display_vocabulary_metrics,
    export_vocabulary,
)

# -----------------------------
# 1. PAGE & SESSION CONFIG
# -----------------------------
st.set_page_config(page_title="Palabrero", page_icon="🌟", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "user_vocabulary" not in st.session_state:
    st.session_state["user_vocabulary"] = set()

if "ai_vocabulary" not in st.session_state:
    st.session_state["ai_vocabulary"] = set()

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = None

if "show_analytics" not in st.session_state:
    st.session_state["show_analytics"] = False

if "message_analytics" not in st.session_state:
    st.session_state["message_analytics"] = []


# -----------------------------
# 2. LOAD SYSTEM PROMPT (.md)
# -----------------------------
def load_system_prompt(file_path: str = "system_prompt.md") -> str:
    """Load the system prompt from a local Markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Initialize system prompt in session state (only once)
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = load_system_prompt("system_prompt.md")


# -----------------------------
# 3. SETUP FUNCTIONS
# -----------------------------
def get_api_key():
    """Securely get the API key from the user."""
    st.sidebar.title("API Key Setup")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        st.session_state["api_key"] = api_key


def initialize_openai_client():
    """Initialize the OpenAI client with the stored API key."""
    if st.session_state["api_key"] and st.session_state["openai_client"] is None:
        st.session_state["openai_client"] = OpenAI(
            api_key=st.session_state["api_key"]
        )


# -----------------------------
# 4. MAIN APP LOGIC
# -----------------------------
def main():
    initialize_database()  
    get_api_key()
    if st.session_state["api_key"] == "":
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

    initialize_openai_client()
    if st.session_state["openai_client"] is None:
        st.warning("OpenAI client could not be initialized.")
        st.stop()

    if st.session_state.get("show_analytics"):
        analytics_dashboard()
        return

    # Display the chat interface and vocabulary metrics
    display_chat()
    display_vocabulary_metrics()

    # User input (chat-style)
    user_input = st.chat_input("Escribe tu mensaje en español")
    if user_input:
        chat_handling(user_input)
        st.rerun()

    # Sidebar for saved chats and analytics
    st.sidebar.header("Session")
    if st.sidebar.button("New Chat"):
        st.session_state["chat_history"] = []
        st.session_state["user_vocabulary"] = set()
        st.session_state["ai_vocabulary"] = set()
        st.session_state["show_analytics"] = False
        st.session_state["message_analytics"] = []
        st.session_state["just_cleared_chat"] = True
        st.rerun()

    if st.session_state.pop("just_cleared_chat", False):
        st.sidebar.info("Started a new chat.")

    # Save chat feature moved to sidebar
    with st.sidebar.expander("Save Current Chat"):
        chat_name = st.text_input("Chat Name:", key="chat_name")
        if st.button("Save Chat"):
            if chat_name.strip():
                save_chat(chat_name.strip())
            else:
                st.sidebar.warning("Please provide a name before saving.")

    st.sidebar.header("Saved Chats")
    saved_chats = load_saved_chats()
    chat_lookup = {chat["name"]: chat for chat in saved_chats}

    def format_chat_option(name: str) -> str:
        if not name:
            return "— Select a chat —"
        meta = chat_lookup.get(name, {})
        saved_at = meta.get("saved_at")
        if saved_at:
            try:
                saved_dt = datetime.datetime.fromisoformat(saved_at)
                return f"{name} · {saved_dt.strftime('%Y-%m-%d %H:%M')}"
            except ValueError:
                return f"{name} · {saved_at}"
        return name

    chat_options = [""] + [chat["name"] for chat in saved_chats]
    selected_chat = st.sidebar.selectbox(
        "Select a chat to load", chat_options, format_func=format_chat_option
    )
    if selected_chat:
        load_chat(selected_chat)
        st.rerun()

    st.sidebar.header("Analytics")
    if st.sidebar.button("Show Analytics Dashboard"):
        st.session_state["show_analytics"] = True
        st.rerun()
    if st.sidebar.button("Export Vocabulary"):
        export_vocabulary()


if __name__ == "__main__":
    main()