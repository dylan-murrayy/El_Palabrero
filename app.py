# Palabrero - A Spanish Language Learning App
# Author: Your Name
# Date: 2023-10-10

import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlite3
import re

# -----------------------------
# 1. PAGE & SESSION CONFIG
# -----------------------------
st.set_page_config(page_title="Palabrero", page_icon="🌟", layout="wide")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'user_vocabulary' not in st.session_state:
    st.session_state['user_vocabulary'] = set()

if 'ai_vocabulary' not in st.session_state:
    st.session_state['ai_vocabulary'] = set()

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''

if 'openai_client' not in st.session_state:
    st.session_state['openai_client'] = None

# -----------------------------
# 2. LOAD SYSTEM PROMPT (.md)
# -----------------------------
def load_system_prompt(file_path="system_prompt.md"):
    """Load the system prompt from a local Markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Initialize system prompt in session state (only once)
if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = load_system_prompt("system_prompt.md")

# -----------------------------
# 3. SETUP FUNCTIONS
# -----------------------------
def get_api_key():
    """Securely get the API key from the user."""
    st.sidebar.title("API Key Setup")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        st.session_state['api_key'] = api_key

def initialize_openai_client():
    """Initialize the OpenAI client with the stored API key."""
    if st.session_state['api_key'] and st.session_state['openai_client'] is None:
        st.session_state['openai_client'] = OpenAI(api_key=st.session_state['api_key'])

def initialize_database():
    """Initialize the local SQLite database and create necessary tables."""
    conn = sqlite3.connect('palabrero.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            name TEXT UNIQUE,
            history TEXT
        )
    ''')
    conn.commit()
    conn.close()

# -----------------------------
# 4. CHAT HANDLER
# -----------------------------
def chat_handling(user_input):
    """Handles the chat flow: sends user input + system message to the model."""
    with st.spinner("Generating AI response..."):
        client = st.session_state['openai_client']
        
        # Prepend system prompt to the messages
        system_message = {
            "role": "system",
            "content": st.session_state['system_prompt']
        }
        
        # Build the messages list: system -> history -> user
        messages = [system_message] + st.session_state['chat_history'] + [
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # Update the model name if needed (e.g., "gpt-3.5-turbo" or "gpt-4")
            messages=messages,
            temperature=0.5,
            max_tokens=150,
        )
    
    ai_message = response.choices[0].message.content
    
    # Update chat history
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_message})

    # Update vocabulary
    update_vocabulary(user_input, ai_message)

    return ai_message

# -----------------------------
# 5. VOCABULARY FUNCTIONS
# -----------------------------
def update_vocabulary(user_text, ai_text):
    """Extract words and update user and AI vocabulary lists."""
    user_words = extract_words(user_text)
    ai_words = extract_words(ai_text)
    st.session_state['user_vocabulary'].update(user_words)
    st.session_state['ai_vocabulary'].update(ai_words)

def extract_words(text):
    """Extract words from text using regex and filter for Spanish."""
    words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
    spanish_words = [word for word in words if is_spanish_word(word)]
    return set(spanish_words)

def is_spanish_word(word):
    """Placeholder function to check if a word is Spanish. 
    Integrate with a dictionary or API for real usage."""
    return True

# -----------------------------
# 6. CHAT DISPLAY
# -----------------------------
def display_chat():
    st.title("Palabrero - Aprende Español")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")

# -----------------------------
# 7. VOCAB METRICS & ANALYTICS
# -----------------------------
def display_vocabulary_metrics():
    st.sidebar.header("Vocabulary Metrics")
    st.sidebar.metric("Active Vocabulary", len(st.session_state['user_vocabulary']))
    st.sidebar.metric("Passive Vocabulary", len(st.session_state['ai_vocabulary']))

def analytics_dashboard():
    st.header("Analytics Dashboard")
    # Create a DataFrame for the user's vocabulary
    dates = [datetime.date.today()] * len(st.session_state['user_vocabulary'])
    vocab_data = pd.DataFrame({
        'Word': list(st.session_state['user_vocabulary']),
        'Date': dates
    })
    st.subheader("Active Vocabulary List")
    st.table(vocab_data)
    # Plot vocabulary growth over time
    st.subheader("Vocabulary Growth Over Time")
    growth_data = vocab_data['Date'].value_counts().sort_index()
    st.line_chart(growth_data)

def export_vocabulary():
    vocab_df = pd.DataFrame({
        'Word': list(st.session_state['user_vocabulary'])
    })
    csv = vocab_df.to_csv(index=False)
    st.download_button("Download Vocabulary as CSV", csv, "vocabulary.csv", "text/csv")

# -----------------------------
# 8. SAVE/LOAD CHAT
# -----------------------------
def save_chat(name):
    """Save the current chat history to the local database."""
    try:
        conn = sqlite3.connect('palabrero.db')
        c = conn.cursor()
        # Create table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                name TEXT UNIQUE,
                history TEXT
            )
        ''')
        # Insert or replace chat history
        c.execute('INSERT OR REPLACE INTO chats (name, history) VALUES (?, ?)', 
                  (name, str(st.session_state['chat_history'])))
        conn.commit()
        conn.close()
        st.success(f"Chat '{name}' saved successfully!")
    except sqlite3.Error as e:
        st.error(f"An error occurred while saving the chat: {e}")

def load_saved_chats():
    """Retrieve a list of saved chat names from the local database."""
    conn = sqlite3.connect('palabrero.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            name TEXT UNIQUE,
            history TEXT
        )
    ''')
    c.execute('SELECT name FROM chats')
    chats = c.fetchall()
    conn.close()
    return [chat[0] for chat in chats]

def load_chat(name):
    """Load a saved chat from the local database by name."""
    try:
        conn = sqlite3.connect('palabrero.db')
        c = conn.cursor()
        c.execute('SELECT history FROM chats WHERE name = ?', (name,))
        history = c.fetchone()
        conn.close()
        if history:
            st.session_state['chat_history'] = eval(history[0])
            st.success(f"Chat '{name}' loaded successfully!")
    except sqlite3.Error as e:
        st.error(f"An error occurred while loading the chat: {e}")

# -----------------------------
# 9. MAIN APP LOGIC
# -----------------------------
def main():
    initialize_database()  
    get_api_key()
    if st.session_state['api_key'] == '':
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

    initialize_openai_client()
    if st.session_state['openai_client'] is None:
        st.warning("OpenAI client could not be initialized.")
        st.stop()

    # Display the chat interface and vocabulary metrics
    display_chat()
    display_vocabulary_metrics()

    # User input
    user_input = st.text_input("Escribe tu mensaje en español:")
    if st.button("Enviar"):
        if user_input:
            ai_response = chat_handling(user_input)
            # Replace st.experimental_rerun() with st.rerun()
            st.rerun()

    # Button to save the current chat
    if st.button("Save Chat"):
        chat_name = st.text_input("Enter a name for your chat:")
        if chat_name:
            save_chat(chat_name)

    # Sidebar for saved chats and analytics
    st.sidebar.header("Saved Chats")
    saved_chats = load_saved_chats()
    selected_chat = st.sidebar.selectbox("Select a chat to load", [""] + saved_chats)
    if selected_chat:
        load_chat(selected_chat)
        # Replace st.experimental_rerun() with st.rerun()
        st.rerun()

    st.sidebar.header("Analytics")
    if st.sidebar.button("Show Analytics Dashboard"):
        analytics_dashboard()
    if st.sidebar.button("Export Vocabulary"):
        export_vocabulary()

# Run the main function
if __name__ == "__main__":
    main()