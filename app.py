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
import json
from collections import Counter
import altair as alt

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

if 'show_analytics' not in st.session_state:
    st.session_state['show_analytics'] = False

if 'message_analytics' not in st.session_state:
    st.session_state['message_analytics'] = []

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
            name TEXT PRIMARY KEY,
            history TEXT,
            saved_at TEXT
        )
    ''')
    try:
        columns = [col[1] for col in c.execute('PRAGMA table_info(chats)')]
        if 'saved_at' not in columns:
            c.execute('ALTER TABLE chats ADD COLUMN saved_at TEXT')
    except sqlite3.OperationalError as err:
        st.warning(f"Unable to ensure database schema: {err}")
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
            temperature=0.8,
            max_tokens=150,
        )
    
    ai_message = response.choices[0].message.content
    
    # Update chat history
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_message})

    # Update vocabulary
    update_vocabulary(user_input, ai_message)

    analysis_result = analyze_user_message(user_input)
    record_message_analysis(user_input, analysis_result)

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

def extract_words(text, unique=True):
    """Extract words from text using regex and filter for Spanish."""
    words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
    spanish_words = [word for word in words if is_spanish_word(word)]
    if not unique:
        return spanish_words
    return set(spanish_words)

def is_spanish_word(word):
    """Placeholder function to check if a word is Spanish. 
    Integrate with a dictionary or API for real usage."""
    return True


def rebuild_vocabulary_from_session():
    """Rebuild vocabulary caches from stored history and analytics."""
    user_vocab = set()
    ai_vocab = set()

    for message in st.session_state.get('chat_history', []):
        if message.get('role') == 'user':
            user_vocab.update(extract_words(message.get('content', '')))
        elif message.get('role') == 'assistant':
            ai_vocab.update(extract_words(message.get('content', '')))

    for entry in st.session_state.get('message_analytics', []):
        analysis = entry.get('analysis') or {}
        for sentence in analysis.get('sentences', []):
            for vocab_item in sentence.get('notable_vocabulary', []):
                word = vocab_item.get('word')
                if word:
                    user_vocab.add(word.lower())

    st.session_state['user_vocabulary'] = user_vocab
    st.session_state['ai_vocabulary'] = ai_vocab


def parse_chat_payload(raw_history):
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


ALLOWED_ERROR_TYPES = [
    "none",
    "grammar",
    "conjugation",
    "agreement",
    "vocabulary",
    "orthography",
    "punctuation",
    "register",
    "pronunciation",
    "other"
]

ALLOWED_TENSES = [
    "present",
    "preterite",
    "imperfect",
    "future",
    "conditional",
    "present-perfect",
    "past-perfect",
    "future-perfect",
    "conditional-perfect",
    "imperative",
    "present-subjunctive",
    "imperfect-subjunctive",
    "other"
]

ALLOWED_TOPICS = [
    "everyday-life",
    "travel",
    "work-and-studies",
    "food-and-cooking",
    "emotions",
    "culture-and-arts",
    "technology",
    "health-and-wellness",
    "relationships",
    "current-events",
    "hobbies",
    "other"
]

ALLOWED_VOCAB_CATEGORIES = [
    "new-word",
    "advanced-word",
    "review-word",
    "idiom",
    "collocation"
]

def analyze_user_message(user_text):
    """Call OpenAI to analyse the learner's message and return structured JSON."""
    client = st.session_state.get('openai_client')
    if not user_text.strip():
        return None
    if client is None:
        st.info("Analítica GPT no disponible. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="no_client")

    analysis_schema = {
        "type": "object",
        "properties": {
            "message_summary": {"type": "string"},
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence_text": {"type": "string"},
                        "detected_tenses": {
                            "type": "array",
                            "items": {"type": "string", "enum": ALLOWED_TENSES},
                            "minItems": 0
                        },
                        "error_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ALLOWED_ERROR_TYPES},
                            "minItems": 0
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string", "enum": ALLOWED_TOPICS},
                            "minItems": 0
                        },
                        "notable_vocabulary": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "word": {"type": "string"},
                                    "category": {"type": "string", "enum": ALLOWED_VOCAB_CATEGORIES},
                                    "english_gloss": {"type": "string"}
                                },
                                "required": ["word", "category"],
                                "additionalProperties": False
                            },
                            "minItems": 0
                        },
                        "feedback": {"type": "string"}
                    },
                    "required": ["sentence_text", "detected_tenses", "error_types", "topics", "notable_vocabulary", "feedback"],
                    "additionalProperties": False
                }
            },
            "overall_error_types": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_ERROR_TYPES},
                "minItems": 0
            },
            "overall_topics": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_TOPICS},
                "minItems": 0
            },
            "key_takeaways": {"type": "string"}
        },
        "required": ["message_summary", "sentences", "overall_error_types", "overall_topics", "key_takeaways"],
        "additionalProperties": False
    }

    system_instructions = (
        "You are an AI Spanish tutor analysing a learner's message. "
        "Split the message into meaningful sentences. "
        "For each sentence, classify verb tenses, error types, topics, and note vocabulary using only the allowed options. "
        "Provide concise feedback sentences in Spanish. "
        "Return strictly valid JSON matching the provided schema. "
        f"Allowed error types: {', '.join(ALLOWED_ERROR_TYPES)}. "
        f"Allowed tenses: {', '.join(ALLOWED_TENSES)}. "
        f"Allowed topics: {', '.join(ALLOWED_TOPICS)}. "
        f"Allowed vocabulary categories: {', '.join(ALLOWED_VOCAB_CATEGORIES)}."
    )

    prompt = f"{system_instructions}\n\nMENSAJE DEL ESTUDIANTE (en español):\n{user_text}"

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "message_analysis",
                    "schema": analysis_schema,
                    "strict": True
                }
            },
            temperature=0.2,
        )
    except Exception as err:
        st.warning(f"No se pudo analizar el mensaje con GPT: {err}")
        return build_fallback_analysis(user_text, reason=str(err))

    analysis_text = None
    # Try the convenience attribute first (if available)
    analysis_text = getattr(response, "output_text", None)

    # Fallback: walk the output structure to collect text content
    if not analysis_text and getattr(response, "output", None):
        chunks = []
        for item in response.output:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if text_value:
                    chunks.append(text_value)
        analysis_text = "".join(chunks).strip() if chunks else None

    if not analysis_text:
        st.info("El modelo no devolvió contenido. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="empty_response")

    try:
        parsed = json.loads(analysis_text)
        if isinstance(parsed, dict):
            parsed.setdefault("source", "gpt")
        return parsed
    except json.JSONDecodeError:
        st.warning("El análisis devuelto no estaba en formato JSON válido. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="json_decode_error")


def record_message_analysis(user_text, analysis_result):
    """Store per-message analytics in session state and update vocab cache."""
    entry = {
        "user_text": user_text,
        "analysis": analysis_result,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    st.session_state['message_analytics'].append(entry)

    if not analysis_result:
        return

    vocab_terms = set()
    for sentence in analysis_result.get("sentences", []):
        for vocab_item in sentence.get("notable_vocabulary", []):
            word = vocab_item.get("word")
            if word:
                vocab_terms.add(word.lower())

    if vocab_terms:
        st.session_state['user_vocabulary'].update(vocab_terms)


def build_fallback_analysis(user_text, reason=None):
    """Provide a heuristic analysis when GPT evaluation is unavailable."""
    cleaned = user_text.strip()
    if not cleaned:
        cleaned = "Mensaje vacío"
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        sentences = [cleaned]

    sentence_entries = []
    for sentence in sentences:
        tokens = extract_words(sentence, unique=False)
        unique_tokens = []
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        vocab_items = [
            {"word": token, "category": "review-word", "english_gloss": ""}
            for token in unique_tokens[:3]
        ]
        sentence_entries.append({
            "sentence_text": sentence,
            "detected_tenses": [],
            "error_types": ["none"],
            "topics": ["other"],
            "notable_vocabulary": vocab_items,
            "feedback": "Buen trabajo. Esta retroalimentación es generada automáticamente."
        })

    analysis = {
        "message_summary": cleaned[:120],
        "sentences": sentence_entries,
        "overall_error_types": ["none"],
        "overall_topics": ["other"],
        "key_takeaways": "Seguimos recopilando datos; esta evaluación se generó con heurísticas.",
        "source": "fallback",
    }
    if reason:
        analysis["fallback_reason"] = reason
    return analysis

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
    st.title("Learning Analytics Dashboard")
    st.caption("Insights generated from GPT-4.1-mini sentence-level evaluations.")

    with st.sidebar:
        if st.button("Back to Chat"):
            st.session_state['show_analytics'] = False
            st.rerun()

    saved_chats = load_saved_chats(include_history=True)
    if not saved_chats:
        st.info("Save at least one conversation to see your analytics.")
        return

    parsed_chats = []
    for chat in saved_chats:
        raw_saved_at = chat.get("saved_at")
        saved_at = None
        if raw_saved_at:
            try:
                saved_at = datetime.datetime.fromisoformat(raw_saved_at)
            except ValueError:
                saved_at = None

        try:
            payload = parse_chat_payload(chat.get("history"))
        except ValueError as err:
            st.warning(f"Skipping chat '{chat.get('name', 'Unknown')}': {err}")
            continue

        parsed_chats.append({
            "name": chat['name'],
            "saved_at": saved_at,
            "raw_saved_at": raw_saved_at,
            "messages": payload.get("messages", []),
            "analytics": payload.get("analytics", [])
        })

    session_analysis_entries = [
        entry for entry in st.session_state.get('message_analytics', [])
        if entry.get("analysis")
    ]
    saved_has_analytics = any(chat["analytics"] for chat in parsed_chats)
    if session_analysis_entries and not saved_has_analytics:
        parsed_chats.append({
            "name": "Current Session (unsaved)",
            "saved_at": datetime.datetime.utcnow(),
            "raw_saved_at": None,
            "messages": st.session_state.get('chat_history', []),
            "analytics": session_analysis_entries,
            "is_session": True
        })

    if not parsed_chats:
        st.info("No analyzable conversations found. Try saving a new chat or send a new message to generate analytics.")
        return

    parsed_chats.sort(key=lambda item: (item["saved_at"] is None, item["saved_at"] or datetime.datetime.min))

    global_vocab = set()
    growth_records = []
    new_words_records = []
    processed_chats = []
    tense_counter = Counter()
    tense_timeline_records = []
    error_counter = Counter()
    topic_counter = Counter()
    vocab_counter = Counter()
    vocab_category_counter = Counter()
    message_records = []
    analysis_available = False
    chats_missing_analysis = []

    for idx, chat in enumerate(parsed_chats, start=1):
        analytics_entries = chat["analytics"]
        conversation_vocab = set()
        conversation_topics = Counter()
        conversation_tenses = Counter()
        conversation_errors = Counter()
        analyzed_messages = 0
        takeaways = []

        for entry in analytics_entries:
            analysis_data = entry.get("analysis")
            if not analysis_data:
                continue
            analysis_source = (analysis_data or {}).get("source", "gpt")
            analysis_available = True
            analyzed_messages += 1

            if analysis_data.get("key_takeaways"):
                takeaways.append(analysis_data["key_takeaways"])

            timestamp_str = entry.get("timestamp")
            message_dt = None
            if timestamp_str:
                try:
                    message_dt = datetime.datetime.fromisoformat(timestamp_str)
                except ValueError:
                    message_dt = None

            sentences = analysis_data.get("sentences", [])
            overall_error_types = analysis_data.get("overall_error_types", [])
            overall_topics = analysis_data.get("overall_topics", [])

            message_records.append({
                "Conversation": chat["name"],
                "Conversation #": idx,
                "Saved At": chat["saved_at"],
                "Message Timestamp": message_dt,
                "Summary": analysis_data.get("message_summary", ""),
                "Overall Errors": ", ".join(overall_error_types) if overall_error_types else "none",
                "Overall Topics": ", ".join(overall_topics) if overall_topics else "none",
                "Source": analysis_data.get("source", "unknown")
            })

            for sentence in sentences:
                detected_tenses = sentence.get("detected_tenses", [])
                error_types = sentence.get("error_types", [])
                topics = sentence.get("topics", [])
                vocab_items = sentence.get("notable_vocabulary", [])

                # Always track the words themselves (for vocabulary size/frequency)
                for vocab_item in vocab_items:
                    word = vocab_item.get("word")
                    if word:
                        word_lower = word.lower()
                        vocab_counter[word_lower] += 1
                        conversation_vocab.add(word_lower)

                # Only treat labels from true GPT analyses as trustworthy for charts
                if analysis_source != "gpt":
                    continue

                for tense in detected_tenses:
                    normalized_tense = (tense or "").strip().lower()
                    if normalized_tense not in ALLOWED_TENSES:
                        normalized_tense = "other"
                    tense_counter[normalized_tense] += 1
                    conversation_tenses[normalized_tense] += 1
                    tense_timeline_records.append({
                        "Conversation #": idx,
                        "Chat": chat["name"],
                        "Saved At": chat["saved_at"],
                        "Message Timestamp": message_dt,
                        "Tense": normalized_tense,
                        "Count": 1
                    })

                for err in error_types:
                    normalized_error = (err or "").strip().lower()
                    if normalized_error == "none":
                        continue
                    if normalized_error not in ALLOWED_ERROR_TYPES:
                        normalized_error = "other"
                    error_counter[normalized_error] += 1
                    conversation_errors[normalized_error] += 1

                for topic in topics:
                    normalized_topic = (topic or "").strip().lower()
                    if normalized_topic == "other":
                        continue
                    if normalized_topic not in ALLOWED_TOPICS:
                        normalized_topic = "other"
                    topic_counter[normalized_topic] += 1
                    conversation_topics[normalized_topic] += 1

                for vocab_item in vocab_items:
                    category = vocab_item.get("category")
                    if not category:
                        continue
                    normalized_category = (category or "").strip().lower()
                    # Skip generic review category so focus areas highlight more informative labels
                    if normalized_category == "review-word":
                        continue
                    if normalized_category not in ALLOWED_VOCAB_CATEGORIES:
                        normalized_category = "other"
                    vocab_category_counter[normalized_category] += 1

        if analyzed_messages == 0:
            chats_missing_analysis.append(chat["name"])

        dominant_tense = conversation_tenses.most_common(1)
        dominant_error = conversation_errors.most_common(1)
        top_topics = [label for label, _ in conversation_topics.most_common(3)]

        new_words = conversation_vocab - global_vocab
        global_vocab.update(conversation_vocab)

        growth_records.append({
            "Saved At": chat["saved_at"],
            "Conversation #": idx,
            "Cumulative Unique Words": len(global_vocab)
        })
        new_words_records.append({
            "Saved At": chat["saved_at"],
            "Conversation #": idx,
            "New Words": len(new_words)
        })

        processed_chats.append({
            "Chat": chat["name"],
            "Saved At": chat["saved_at"],
            "Conversation #": idx,
            "Analyzed Messages": analyzed_messages,
            "Unique Vocabulary": len(conversation_vocab),
            "Dominant Tense": dominant_tense[0][0] if dominant_tense else "N/A",
            "Top Error Type": dominant_error[0][0] if dominant_error else "N/A",
            "Primary Topics": ", ".join(top_topics) if top_topics else "N/A",
            "Latest Takeaway": takeaways[-1] if takeaways else "",
            "Source": "Mixta" if any(
                (entry.get("analysis") or {}).get("source") != "gpt" for entry in analytics_entries
            ) else "GPT"
        })

    if not analysis_available:
        st.info("No GPT-powered analytics found yet. Send new messages to generate insights.")
        return

    summary_df = pd.DataFrame(processed_chats)
    summary_df['Saved At Display'] = summary_df['Saved At'].apply(
        lambda dt: dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime.datetime) and pd.notna(dt) else "Not recorded"
    )

    growth_df = pd.DataFrame(growth_records)
    new_words_df = pd.DataFrame(new_words_records)
    tense_overall_df = pd.DataFrame(
        [{"Tense": tense, "Count": count} for tense, count in tense_counter.items()]
    ).sort_values("Count", ascending=False) if tense_counter else pd.DataFrame()
    error_overall_df = pd.DataFrame(
        [{"Error Type": error, "Count": count} for error, count in error_counter.items()]
    ).sort_values("Count", ascending=False) if error_counter else pd.DataFrame()
    topic_overall_df = pd.DataFrame(
        [{"Topic": topic, "Count": count} for topic, count in topic_counter.items()]
    ).sort_values("Count", ascending=False) if topic_counter else pd.DataFrame()
    vocab_category_df = pd.DataFrame(
        [{"Category": category, "Count": count} for category, count in vocab_category_counter.items()]
    ).sort_values("Count", ascending=False) if vocab_category_counter else pd.DataFrame()
    top_words_df = pd.DataFrame(
        [{"Word": word, "Frequency": freq} for word, freq in vocab_counter.most_common(20)]
    ) if vocab_counter else pd.DataFrame()
    tense_timeline_df = pd.DataFrame(tense_timeline_records)
    message_detail_df = pd.DataFrame(message_records)

    total_analyzed_messages = int(summary_df['Analyzed Messages'].sum())
    fallback_rows = summary_df[summary_df['Source'] != "GPT"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Conversations", len(summary_df))
    col2.metric("Analyzed Messages", total_analyzed_messages)
    col3.metric("Unique Vocabulary (GPT)", len(global_vocab))

    top_tense = tense_counter.most_common(1)
    top_error = error_counter.most_common(1)
    top_topic = topic_counter.most_common(1)

    col4, col5, col6 = st.columns(3)
    col4.metric("Most Used Tense", f"{top_tense[0][0]} ({top_tense[0][1]})" if top_tense else "N/A")
    col5.metric("Top Error Type", f"{top_error[0][0]} ({top_error[0][1]})" if top_error else "N/A")
    col6.metric("Top Topic", f"{top_topic[0][0]} ({top_topic[0][1]})" if top_topic else "N/A")

    if not fallback_rows.empty:
        st.info(
            "Algunas conversaciones usan análisis heurístico porque la llamada a GPT falló o no está disponible."
        )

    st.markdown("---")

    has_timestamp = not growth_df['Saved At'].dropna().empty

    if not growth_df.empty:
        if has_timestamp:
            growth_chart_df = growth_df.dropna(subset=['Saved At']).copy()
            growth_chart_df['Saved At'] = pd.to_datetime(growth_chart_df['Saved At'])
            growth_chart_df['Saved Date'] = growth_chart_df['Saved At'].dt.normalize()
            daily_growth_df = (
                growth_chart_df
                .groupby('Saved Date', as_index=False)['Cumulative Unique Words']
                .max()
            )
            growth_chart = alt.Chart(daily_growth_df).mark_line(point=True).encode(
                x=alt.X('Saved Date:T', title='Date'),
                y=alt.Y('Cumulative Unique Words:Q', title='Cumulative Unique Words'),
                tooltip=['Saved Date:T', 'Cumulative Unique Words:Q']
            ).properties(height=300)
        else:
            growth_chart = alt.Chart(growth_df).mark_line(point=True).encode(
                x=alt.X('Conversation #:Q', title='Conversation'),
                y=alt.Y('Cumulative Unique Words:Q', title='Cumulative Unique Words'),
                tooltip=['Conversation #:Q', 'Cumulative Unique Words:Q']
            ).properties(height=300)
        st.subheader("Vocabulary Growth")
        st.altair_chart(growth_chart, use_container_width=True)

    if not new_words_df.empty:
        if has_timestamp:
            new_words_chart_df = new_words_df.dropna(subset=['Saved At']).copy()
            new_words_chart_df['Saved At'] = pd.to_datetime(new_words_chart_df['Saved At'])
            new_words_chart_df['Saved Date'] = new_words_chart_df['Saved At'].dt.normalize()
            daily_new_words_df = (
                new_words_chart_df
                .groupby('Saved Date', as_index=False)['New Words']
                .sum()
            )
            new_words_chart = alt.Chart(daily_new_words_df).mark_bar().encode(
                x=alt.X('Saved Date:T', title='Date'),
                y=alt.Y('New Words:Q', title='New Words Introduced'),
                tooltip=['Saved Date:T', 'New Words:Q']
            ).properties(height=250)
        else:
            new_words_chart = alt.Chart(new_words_df).mark_bar().encode(
                x=alt.X('Conversation #:Q', title='Conversation'),
                y=alt.Y('New Words:Q', title='New Words Introduced'),
                tooltip=['Conversation #:Q', 'New Words:Q']
            ).properties(height=250)
        st.subheader("New Vocabulary Introduced")
        st.altair_chart(new_words_chart, use_container_width=True)

    if not tense_overall_df.empty:
        st.subheader("Verb Tense Usage")
        tense_chart = alt.Chart(tense_overall_df).mark_bar().encode(
            x=alt.X('Count:Q', title='Occurrences'),
            y=alt.Y('Tense:N', sort='-x', title='Verb Tense'),
            tooltip=['Tense:N', 'Count:Q']
        ).properties(height=300)
        st.altair_chart(tense_chart, use_container_width=True)

        if not tense_timeline_df.empty:
            if tense_timeline_df['Message Timestamp'].notna().any():
                timeline_df = tense_timeline_df.dropna(subset=['Message Timestamp']).copy()
                timeline_df['Message Timestamp'] = pd.to_datetime(timeline_df['Message Timestamp'])
                timeline_chart = alt.Chart(timeline_df).mark_line(point=True).encode(
                    x=alt.X('Message Timestamp:T', title='Message Timestamp'),
                    y=alt.Y('Count:Q', aggregate='sum', title='Occurrences'),
                    color=alt.Color('Tense:N', title='Verb Tense'),
                    tooltip=['Message Timestamp:T', 'Tense:N', 'Count:Q']
                ).properties(height=320)
            else:
                timeline_chart = alt.Chart(tense_timeline_df).mark_line(point=True).encode(
                    x=alt.X('Conversation #:Q', title='Conversation'),
                    y=alt.Y('Count:Q', aggregate='sum', title='Occurrences'),
                    color=alt.Color('Tense:N', title='Verb Tense'),
                    tooltip=['Conversation #:Q', 'Tense:N', 'Count:Q']
                ).properties(height=320)
            st.altair_chart(timeline_chart, use_container_width=True)

    if not error_overall_df.empty:
        st.subheader("Error Type Distribution")
        error_chart = alt.Chart(error_overall_df).mark_bar(color="#E4572E").encode(
            x=alt.X('Count:Q', title='Occurrences'),
            y=alt.Y('Error Type:N', sort='-x', title='Error Type'),
            tooltip=['Error Type:N', 'Count:Q']
        ).properties(height=280)
        st.altair_chart(error_chart, use_container_width=True)

    if not topic_overall_df.empty:
        st.subheader("Topic Coverage")
        topic_chart = alt.Chart(topic_overall_df).mark_bar(color="#17BEBB").encode(
            x=alt.X('Count:Q', title='Mentions'),
            y=alt.Y('Topic:N', sort='-x', title='Topic'),
            tooltip=['Topic:N', 'Count:Q']
        ).properties(height=280)
        st.altair_chart(topic_chart, use_container_width=True)

    if not vocab_category_df.empty:
        st.subheader("Vocabulary Focus Areas")
        vocab_category_chart = alt.Chart(vocab_category_df).mark_bar(color="#FFC914").encode(
            x=alt.X('Count:Q', title='Occurrences'),
            y=alt.Y('Category:N', sort='-x', title='Category'),
            tooltip=['Category:N', 'Count:Q']
        ).properties(height=240)
        st.altair_chart(vocab_category_chart, use_container_width=True)

    if not top_words_df.empty:
        st.subheader("Top Vocabulary (by frequency)")
        top_words_chart = alt.Chart(top_words_df).mark_bar().encode(
            x=alt.X('Frequency:Q', title='Frequency'),
            y=alt.Y('Word:N', sort='-x', title='Word'),
            tooltip=['Word:N', 'Frequency:Q']
        ).properties(height=320)
        st.altair_chart(top_words_chart, use_container_width=True)

    st.subheader("Conversation Summary")
    display_df = summary_df[['Chat', 'Saved At Display', 'Source', 'Analyzed Messages', 'Unique Vocabulary',
                             'Dominant Tense', 'Top Error Type', 'Primary Topics', 'Latest Takeaway']].rename(
        columns={'Saved At Display': 'Saved At'}
    )
    st.dataframe(display_df, use_container_width=True)

    if not message_detail_df.empty:
        st.subheader("Message-Level Details")
        detail_view = message_detail_df.copy()
        detail_view['Saved At'] = detail_view['Saved At'].apply(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime.datetime) and pd.notna(dt) else "Not recorded"
        )
        detail_view['Message Timestamp'] = detail_view['Message Timestamp'].apply(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime.datetime) and pd.notna(dt) else "Not recorded"
        )
        st.dataframe(detail_view, use_container_width=True)

    st.markdown("---")
    aggregated_vocab = sorted(global_vocab)
    if aggregated_vocab:
        vocab_df = pd.DataFrame({'Word': aggregated_vocab})
        csv_bytes = vocab_df.to_csv(index=False)
        st.subheader("Download Your GPT-Derived Vocabulary")
        st.download_button(
            "Download Vocabulary as CSV",
            csv_bytes,
            "palabrero_vocabulary.csv",
            "text/csv"
        )

    if chats_missing_analysis:
        st.info(
            "These chats do not yet have GPT analytics and were skipped: "
            + ", ".join(chats_missing_analysis)
        )

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
                name TEXT PRIMARY KEY,
                history TEXT,
                saved_at TEXT
            )
        ''')
        payload = {
            "messages": st.session_state['chat_history'],
            "analytics": st.session_state.get('message_analytics', [])
        }
        history_json = json.dumps(payload)
        saved_at = datetime.datetime.utcnow().isoformat()
        c.execute('INSERT OR REPLACE INTO chats (name, history, saved_at) VALUES (?, ?, ?)', 
                  (name, history_json, saved_at))
        conn.commit()
        conn.close()
        st.success(f"Chat '{name}' saved successfully!")
    except sqlite3.Error as e:
        st.error(f"An error occurred while saving the chat: {e}")

def load_saved_chats(include_history=False):
    """Retrieve saved chat metadata (and optionally full history) from the local database."""
    conn = sqlite3.connect('palabrero.db')
    c = conn.cursor()
    order_clause = '''
        ORDER BY 
            CASE WHEN saved_at IS NULL THEN 1 ELSE 0 END,
            saved_at DESC,
            name ASC
    '''
    if include_history:
        c.execute(f'SELECT name, history, saved_at FROM chats {order_clause}')
        rows = c.fetchall()
        chats = [
            {"name": row[0], "history": row[1], "saved_at": row[2]}
            for row in rows
        ]
    else:
        c.execute(f'SELECT name, saved_at FROM chats {order_clause}')
        rows = c.fetchall()
        chats = [
            {"name": row[0], "saved_at": row[1]}
            for row in rows
        ]
    conn.close()
    return chats

def load_chat(name):
    """Load a saved chat from the local database by name."""
    try:
        conn = sqlite3.connect('palabrero.db')
        c = conn.cursor()
        c.execute('SELECT history FROM chats WHERE name = ?', (name,))
        history = c.fetchone()
        conn.close()
        if history and history[0]:
            try:
                parsed = parse_chat_payload(history[0])
            except ValueError as err:
                st.error(f"Failed to load chat '{name}': {err}")
            else:
                st.session_state['chat_history'] = parsed.get("messages", [])
                st.session_state['message_analytics'] = parsed.get("analytics", [])
                st.session_state['show_analytics'] = False
                rebuild_vocabulary_from_session()
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

    if st.session_state.get('show_analytics'):
        analytics_dashboard()
        return

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
    chat_name = st.text_input("Enter a name for your chat:", key="chat_name")
    if st.button("Save Chat"):
        if chat_name.strip():
            save_chat(chat_name.strip())
        else:
            st.warning("Please provide a name before saving your chat.")

    # Sidebar for saved chats and analytics
    st.sidebar.header("Session")
    if st.sidebar.button("New Chat"):
        st.session_state['chat_history'] = []
        st.session_state['user_vocabulary'] = set()
        st.session_state['ai_vocabulary'] = set()
        st.session_state['show_analytics'] = False
        st.session_state['message_analytics'] = []
        st.session_state['just_cleared_chat'] = True
        st.rerun()

    if st.session_state.pop('just_cleared_chat', False):
        st.sidebar.info("Started a new chat.")

    st.sidebar.header("Saved Chats")
    saved_chats = load_saved_chats()
    chat_lookup = {chat['name']: chat for chat in saved_chats}

    def format_chat_option(name):
        if not name:
            return "— Select a chat —"
        meta = chat_lookup.get(name, {})
        saved_at = meta.get('saved_at')
        if saved_at:
            try:
                saved_dt = datetime.datetime.fromisoformat(saved_at)
                return f"{name} · {saved_dt.strftime('%Y-%m-%d %H:%M')}"
            except ValueError:
                return f"{name} · {saved_at}"
        return name

    chat_options = [""] + [chat['name'] for chat in saved_chats]
    selected_chat = st.sidebar.selectbox("Select a chat to load", chat_options, format_func=format_chat_option)
    if selected_chat:
        load_chat(selected_chat)
        # Replace st.experimental_rerun() with st.rerun()
        st.rerun()

    st.sidebar.header("Analytics")
    if st.sidebar.button("Show Analytics Dashboard"):
        st.session_state['show_analytics'] = True
        st.rerun()
    if st.sidebar.button("Export Vocabulary"):
        export_vocabulary()

# Run the main function
if __name__ == "__main__":
    main()