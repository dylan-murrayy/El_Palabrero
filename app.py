import datetime

import streamlit as st
from openai import OpenAI

from chat import get_chat_stream, process_chat_turn
from storage import initialize_database, load_chat, load_saved_chats, save_chat, delete_chats
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
    with st.sidebar.expander("Settings", expanded=not st.session_state.get("api_key")):
        api_key = st.text_input("OpenAI API Key:", type="password", value=st.session_state.get("api_key", ""))
        if api_key:
            st.session_state["api_key"] = api_key
        
        # TTS Auto-play toggle
        st.session_state["tts_auto_play"] = st.checkbox(
            "Auto-play audio for new AI messages",
            value=st.session_state.get("tts_auto_play", False),
            help="Automatically generate and play audio when AI responds"
        )


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
    # Inject custom CSS
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Gracefully handle missing CSS file
    
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
        # 1. Display user message immediately
        with st.chat_message("user", avatar="🫠"):
            st.markdown(user_input)
        
        # 2. Stream assistant response
        with st.chat_message("assistant", avatar="🤖"):
            stream = get_chat_stream(user_input)
            ai_response = st.write_stream(stream)
        
        # 3. Update history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
        
        # 4. Post-processing (analytics, vocab)
        process_chat_turn(user_input, ai_response)
        
        # 5. Rerun to update state/sidebar
        st.rerun()

    # Sidebar Navigation
    st.sidebar.title("Palabrero")
    
    # 1. VIEW MODE
    st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)
    view_mode = st.sidebar.radio(
        "View",
        ["Chat", "Analytics"],
        index=0 if not st.session_state.get("show_analytics") else 1,
        horizontal=False,
        label_visibility="collapsed"
    )
    
    # Update show_analytics based on selection
    if view_mode == "Analytics" and not st.session_state.get("show_analytics"):
        st.session_state["show_analytics"] = True
        st.rerun()
    elif view_mode == "Chat" and st.session_state.get("show_analytics"):
        st.session_state["show_analytics"] = False
        st.rerun()
    
    # 2. SESSION MANAGEMENT
    st.sidebar.markdown('<p class="sidebar-header">Session</p>', unsafe_allow_html=True)
    
    col_new, col_save = st.sidebar.columns([1, 1])
    with col_new:
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state["chat_history"] = []
            st.session_state["user_vocabulary"] = set()
            st.session_state["ai_vocabulary"] = set()
            st.session_state["show_analytics"] = False
            st.session_state["message_analytics"] = []
            st.session_state["just_cleared_chat"] = True
            st.rerun()
            
    with st.sidebar.expander("Save Chat", expanded=False):
        chat_name = st.text_input("Name", placeholder="My Conversation", key="chat_name")
        if st.button("Save", use_container_width=True):
            if chat_name.strip():
                save_chat(chat_name.strip())
            else:
                st.warning("Please provide a name.")

    if st.session_state.pop("just_cleared_chat", False):
        st.toast("Started a new chat!", icon="✨")
    
    # 3. HISTORY
    st.sidebar.markdown('<p class="sidebar-header">History</p>', unsafe_allow_html=True)
    saved_chats = load_saved_chats()
    chat_lookup = {chat["name"]: chat for chat in saved_chats}

    def format_chat_option(name: str) -> str:
        if not name:
            return "Select a chat..."
        meta = chat_lookup.get(name, {})
        saved_at = meta.get("saved_at")
        if saved_at:
            try:
                saved_dt = datetime.datetime.fromisoformat(saved_at)
                return f"{name} ({saved_dt.strftime('%m/%d')})"
            except ValueError:
                return name
        return name

    chat_options = [""] + [chat["name"] for chat in saved_chats]
    selected_chat = st.sidebar.selectbox(
        "Load Chat", chat_options, format_func=format_chat_option,
        label_visibility="collapsed"
    )
    if selected_chat:
        load_chat(selected_chat)
        st.rerun()
    
    # Delete Conversations
    if saved_chats:
        with st.sidebar.expander("Manage Chats"):
            chats_to_delete = st.multiselect(
                "Select to delete",
                [chat["name"] for chat in saved_chats],
                key="delete_chats_selector"
            )
            
            if st.button("Delete Selected", type="secondary", use_container_width=True, disabled=not chats_to_delete):
                delete_chats(chats_to_delete)
                st.rerun()
    
    # 4. VOCABULARY (Only in Chat view)
    if not st.session_state.get("show_analytics"):
        st.sidebar.markdown('<p class="sidebar-header">Vocabulary</p>', unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Active", len(st.session_state["user_vocabulary"]))
        col2.metric("Passive", len(st.session_state["ai_vocabulary"]))
        
        if st.sidebar.button("Export CSV", use_container_width=True):
            export_vocabulary()


if __name__ == "__main__":
    main()