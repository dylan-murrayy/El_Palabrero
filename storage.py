import datetime
import json
import sqlite3
import streamlit as st

from vocabulary import rebuild_vocabulary_from_session


def initialize_database():
    """Initialize the local SQLite database and create necessary tables."""
    conn = sqlite3.connect("palabrero.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            name TEXT PRIMARY KEY,
            history TEXT,
            saved_at TEXT
        )
        """
    )
    try:
        columns = [col[1] for col in c.execute("PRAGMA table_info(chats)")]
        if "saved_at" not in columns:
            c.execute("ALTER TABLE chats ADD COLUMN saved_at TEXT")
    except sqlite3.OperationalError as err:
        st.warning(f"Unable to ensure database schema: {err}")
    conn.commit()
    conn.close()


def save_chat(name: str):
    """Save the current chat history to the local database."""
    try:
        conn = sqlite3.connect("palabrero.db")
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                name TEXT PRIMARY KEY,
                history TEXT,
                saved_at TEXT
            )
            """
        )
        payload = {
            "messages": st.session_state["chat_history"],
            "analytics": st.session_state.get("message_analytics", []),
        }
        history_json = json.dumps(payload)
        saved_at = datetime.datetime.utcnow().isoformat()
        c.execute(
            "INSERT OR REPLACE INTO chats (name, history, saved_at) VALUES (?, ?, ?)",
            (name, history_json, saved_at),
        )
        conn.commit()
        conn.close()
        st.success(f"Chat '{name}' saved successfully!")
    except sqlite3.Error as err:
        st.error(f"An error occurred while saving the chat: {err}")


def load_saved_chats(include_history: bool = False):
    """Retrieve saved chat metadata (and optionally full history) from the local database."""
    conn = sqlite3.connect("palabrero.db")
    c = conn.cursor()
    order_clause = """
        ORDER BY 
            CASE WHEN saved_at IS NULL THEN 1 ELSE 0 END,
            saved_at DESC,
            name ASC
    """
    if include_history:
        c.execute(f"SELECT name, history, saved_at FROM chats {order_clause}")
        rows = c.fetchall()
        chats = [{"name": row[0], "history": row[1], "saved_at": row[2]} for row in rows]
    else:
        c.execute(f"SELECT name, saved_at FROM chats {order_clause}")
        rows = c.fetchall()
        chats = [{"name": row[0], "saved_at": row[1]} for row in rows]
    conn.close()
    return chats


def parse_chat_payload(raw_history: str | None):
    """Parse stored chat data into messages and analytics."""
    if not raw_history:
        return {"messages": [], "analytics": []}
    try:
        data = json.loads(raw_history)
    except json.JSONDecodeError as err:
        raise ValueError(f"Stored chat history is not valid JSON: {err}") from err
    if isinstance(data, dict):
        messages = data.get("messages")
        analytics = data.get("analytics", [])
        if isinstance(messages, list):
            if not isinstance(analytics, list):
                analytics = []
            return {"messages": messages, "analytics": analytics}
        raise ValueError("Stored chat structure is missing 'messages' list.")
    if isinstance(data, list):
        return {"messages": data, "analytics": []}
    raise ValueError("Stored chat history format is not supported.")


def load_chat(name: str):
    """Load a saved chat from the local database by name."""
    try:
        conn = sqlite3.connect("palabrero.db")
        c = conn.cursor()
        c.execute("SELECT history FROM chats WHERE name = ?", (name,))
        history = c.fetchone()
        conn.close()
        if history and history[0]:
            try:
                parsed = parse_chat_payload(history[0])
            except ValueError as err:
                st.error(f"Failed to load chat '{name}': {err}")
            else:
                st.session_state["chat_history"] = parsed.get("messages", [])
                st.session_state["message_analytics"] = parsed.get("analytics", [])
                st.session_state["show_analytics"] = False
                rebuild_vocabulary_from_session()
            st.success(f"Chat '{name}' loaded successfully!")
    except sqlite3.Error as err:
        st.error(f"An error occurred while loading the chat: {err}")


