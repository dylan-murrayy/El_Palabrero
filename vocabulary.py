import re
import streamlit as st


def extract_words(text, unique: bool = True):
    """Extract words from text using regex and filter for Spanish."""
    words = re.findall(r"\b\w+\b", text.lower(), re.UNICODE)
    spanish_words = [word for word in words if is_spanish_word(word)]
    if not unique:
        return spanish_words
    return set(spanish_words)


def is_spanish_word(word: str) -> bool:
    """Placeholder function to check if a word is Spanish.

    Integrate with a dictionary or API for real usage.
    """
    return True


def update_vocabulary(user_text: str, ai_text: str):
    """Extract words and update user and AI vocabulary lists."""
    user_words = extract_words(user_text)
    ai_words = extract_words(ai_text)
    st.session_state["user_vocabulary"].update(user_words)
    st.session_state["ai_vocabulary"].update(ai_words)


def rebuild_vocabulary_from_session():
    """Rebuild vocabulary caches from stored history and analytics."""
    user_vocab = set()
    ai_vocab = set()

    for message in st.session_state.get("chat_history", []):
        if message.get("role") == "user":
            user_vocab.update(extract_words(message.get("content", "")))
        elif message.get("role") == "assistant":
            ai_vocab.update(extract_words(message.get("content", "")))

    for entry in st.session_state.get("message_analytics", []):
        analysis = entry.get("analysis") or {}
        for sentence in analysis.get("sentences", []):
            for vocab_item in sentence.get("notable_vocabulary", []):
                word = vocab_item.get("word")
                if word:
                    user_vocab.add(word.lower())

    st.session_state["user_vocabulary"] = user_vocab
    st.session_state["ai_vocabulary"] = ai_vocab


