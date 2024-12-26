# Palabrero - A Spanish Language Learning App
# Author: Your Name
# Date: 2023-10-10

import streamlit as st
from openai import OpenAI  # Updated import statement
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlite3
import re

# Set page configuration
st.set_page_config(page_title="Palabrero", page_icon="🌟", layout="wide")

# Initialize session state variables
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

# Function to securely get the API key from the user
def get_api_key():
    st.sidebar.title("API Key Setup")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        st.session_state['api_key'] = api_key

# Function to initialize the OpenAI client
def initialize_openai_client():
    if st.session_state['api_key'] and st.session_state['openai_client'] is None:
        st.session_state['openai_client'] = OpenAI(api_key=st.session_state['api_key'])

# Function to initialize the database and create tables
def initialize_database():
    conn = sqlite3.connect('palabrero.db')
    c = conn.cursor()
    # Create chats table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            name TEXT UNIQUE,
            history TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to handle chat interactions with the AI
def chat_handling(user_input):
    # Display a loading spinner while generating the AI response
    with st.spinner("Generating AI response..."):
        client = st.session_state['openai_client']
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state['chat_history'] + [{"role": "user", "content": user_input}],
            temperature=0.5,
            max_tokens=150,
        )
    ai_message = response.choices[0].message.content
    # Update chat history
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_message})
    # Update vocabulary tracking
    update_vocabulary(user_input, ai_message)
    return ai_message

# Function to update active and passive vocabulary lists
def update_vocabulary(user_text, ai_text):
    user_words = extract_words(user_text)
    ai_words = extract_words(ai_text)
    st.session_state['user_vocabulary'].update(user_words)
    st.session_state['ai_vocabulary'].update(ai_words)

# Function to extract words from text using regex
def extract_words(text):
    words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
    spanish_words = [word for word in words if is_spanish_word(word)]
    return set(spanish_words)

# Placeholder function to check if a word is Spanish
def is_spanish_word(word):
    # For simplicity, assume all words are Spanish
    # Integrate with a Spanish dictionary API or word list for real implementation
    return True

# Function to display the chat interface
def display_chat():
    st.title("Palabrero - Aprende Español")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")

# Function to display live vocabulary metrics
def display_vocabulary_metrics():
    st.sidebar.header("Vocabulary Metrics")
    st.sidebar.metric("Active Vocabulary", len(st.session_state['user_vocabulary']))
    st.sidebar.metric("Passive Vocabulary", len(st.session_state['ai_vocabulary']))

# Function to display the analytics dashboard
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

# Function to export vocabulary data as CSV
def export_vocabulary():
    vocab_df = pd.DataFrame({
        'Word': list(st.session_state['user_vocabulary'])
    })
    csv = vocab_df.to_csv(index=False)
    st.download_button("Download Vocabulary as CSV", csv, "vocabulary.csv", "text/csv")

# Function to save the current chat history with a custom name
def save_chat(name):
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
        c.execute('INSERT OR REPLACE INTO chats (name, history) VALUES (?, ?)', (name, str(st.session_state['chat_history'])))
        conn.commit()
        conn.close()
        st.success(f"Chat '{name}' saved successfully!")
    except sqlite3.Error as e:
        st.error(f"An error occurred while saving the chat: {e}")

# Function to retrieve a list of saved chats
def load_saved_chats():
    conn = sqlite3.connect('palabrero.db')
    c = conn.cursor()
    # Ensure the table exists
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

# Function to load a saved chat by name
def load_chat(name):
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

# Main function to orchestrate the app logic
def main():
    initialize_database()  # Initialize the database at startup
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

    # User input area for sending messages
    user_input = st.text_input("Escribe tu mensaje en español:")
    if st.button("Enviar"):
        if user_input:
            ai_response = chat_handling(user_input)
            st.rerun()  # Updated function call

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
        st.rerun()  # Updated function call

    st.sidebar.header("Analytics")
    if st.sidebar.button("Show Analytics Dashboard"):
        analytics_dashboard()
    if st.sidebar.button("Export Vocabulary"):
        export_vocabulary()



# Run the main function when the script is executed
if __name__ == "__main__":
    main()
