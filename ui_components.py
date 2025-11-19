import datetime
from collections import Counter
import json
import os
from typing import List, Dict, Any

import altair as alt
import pandas as pd
import streamlit as st

from flashcards import generate_cloze_cards, create_mochi_zip

SCENARIOS_FILE = "scenarios.json"

def load_scenarios() -> List[Dict[str, Any]]:
    """Load scenarios from JSON file."""
    if not os.path.exists(SCENARIOS_FILE):
        return []
    try:
        with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_scenarios(scenarios: List[Dict[str, Any]]):
    """Save scenarios to JSON file."""
    with open(SCENARIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=4, ensure_ascii=False)

def display_scenario_manager():
    """Render the scenario selector and manager in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Roleplay Scenario")
    
    scenarios = load_scenarios()
    if not scenarios:
        # Fallback if file missing
        scenarios = [{"id": "default", "name": "Standard Tutor", "description": "Default mode", "system_prompt": None}]
    
    # Selection
    scenario_names = [s["name"] for s in scenarios]
    
    # Determine current index
    current_id = st.session_state.get("selected_scenario_id", "default")
    try:
        current_index = next(i for i, s in enumerate(scenarios) if s["id"] == current_id)
    except StopIteration:
        current_index = 0
        
    selected_name = st.sidebar.selectbox(
        "Choose a Scenario",
        scenario_names,
        index=current_index,
        key="scenario_selector"
    )
    
    # Update session state if changed
    selected_scenario = next(s for s in scenarios if s["name"] == selected_name)
    if st.session_state.get("selected_scenario_id") != selected_scenario["id"]:
        st.session_state["selected_scenario_id"] = selected_scenario["id"]
        st.rerun()

    st.sidebar.caption(selected_scenario.get("description", ""))
    
    # Manager (Expander)
    with st.sidebar.expander("Manage Scenarios"):
        # Add New
        with st.form("add_scenario_form"):
            st.write("Create New Scenario")
            new_name = st.text_input("Name")
            new_desc = st.text_input("Description")
            new_prompt = st.text_area("System Prompt (Instructions for AI)")
            submitted = st.form_submit_button("Add Scenario")
            
            if submitted and new_name and new_prompt:
                new_id = new_name.lower().replace(" ", "_")
                new_scenario = {
                    "id": new_id,
                    "name": new_name,
                    "description": new_desc,
                    "system_prompt": new_prompt
                }
                scenarios.append(new_scenario)
                save_scenarios(scenarios)
                st.success("Scenario added!")
                st.rerun()
        
        # Delete (only custom ones)
        custom_scenarios = [s for s in scenarios if s["id"] != "default"]
        if custom_scenarios:
            st.write("Delete Scenario")
            to_delete = st.selectbox("Select to delete", [s["name"] for s in custom_scenarios], key="delete_scenario_select")
            if st.button("Delete Selected"):
                scenarios = [s for s in scenarios if s["name"] != to_delete]
                save_scenarios(scenarios)
                # Reset to default if deleted current
                if st.session_state.get("selected_scenario_id") == next((s["id"] for s in custom_scenarios if s["name"] == to_delete), ""):
                     st.session_state["selected_scenario_id"] = "default"
                st.success("Deleted.")
                st.rerun()

from analysis import (
    ALLOWED_ERROR_TYPES,
    ALLOWED_TENSES,
    ALLOWED_TOPICS,
    ALLOWED_VOCAB_CATEGORIES,
)
from storage import load_saved_chats, parse_chat_payload


def generate_tts_audio(text: str, voice: str = "alloy") -> bytes | None:
    """
    Generate text-to-speech audio using OpenAI's TTS API.
    
    Args:
        text: The text to convert to speech
        voice: The voice to use (alloy, echo, fable, onyx, nova, shimmer)
    
    Returns:
        Audio bytes in MP3 format, or None if generation fails
    """
    client = st.session_state.get("openai_client")
    if not client:
        return None
    
    if not text or not text.strip():
        return None
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        return response.content
    except Exception as err:  # noqa: BLE001
        st.warning(f"Error generating speech: {err}")
        return None


def display_chat():
    """Render the main chat conversation."""
    st.title("Palabrero - Aprende Español")
    history = st.session_state.get("chat_history", [])

    # First-run / empty state with a friendly onboarding message
    if not history:
        st.markdown("""
        <div class="dashboard-card" style="text-align: center; padding: 3rem 1rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">👋</h1>
            <h2>¡Bienvenido a Palabrero!</h2>
            <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto 2rem auto;">
                Tu compañero de conversación inteligente. Escribe abajo para empezar a practicar.
                Te ayudaré con correcciones, vocabulario y gramática.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(99, 102, 241, 0.1); color: #818cf8; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                    ✨ Correcciones en tiempo real
                </span>
                <span style="background: rgba(16, 185, 129, 0.1); color: #34d399; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                    📊 Análisis de progreso
                </span>
                <span style="background: rgba(244, 63, 94, 0.1); color: #fb7185; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                    🗣️ Práctica de conversación
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Render chat history using Streamlit's native chat UI
    for idx, message in enumerate(history):
        role = message.get("role", "assistant")
        content = message.get("content", "")
        # Map roles to Streamlit chat roles and simple avatars
        if role == "user":
            name = "user"
            avatar = "🫠"
        else:
            name = "assistant"
            avatar = "🤖"

        with st.chat_message(name, avatar=avatar):
            st.markdown(content)
            
            # Add TTS button for AI messages
            if role == "assistant" and content.strip():
                # Create a unique key for each message's TTS button and audio storage
                tts_key = f"tts_button_{idx}"
                tts_audio_key = f"tts_audio_{idx}"
                tts_generate_key = f"tts_generate_{idx}"
                tts_autoplay_key = f"tts_autoplay_{idx}"
                
                # Initialize audio storage if not exists
                if tts_audio_key not in st.session_state:
                    st.session_state[tts_audio_key] = None
                if tts_generate_key not in st.session_state:
                    st.session_state[tts_generate_key] = False
                if tts_autoplay_key not in st.session_state:
                    st.session_state[tts_autoplay_key] = False
                
                # Compact button using columns
                col1, col2, col3 = st.columns([1, 10, 1])
                with col1:
                    if st.button("🔊", key=tts_key, help="Generate and play audio"):
                        st.session_state[tts_generate_key] = True
                        st.rerun()
                
                # Generate audio if requested or auto-play is enabled for new messages
                auto_play_enabled = st.session_state.get("tts_auto_play", False)
                is_latest_message = (idx == len(history) - 1)
                
                if st.session_state.get(tts_generate_key) or (auto_play_enabled and is_latest_message and st.session_state.get(tts_audio_key) is None):
                    with st.spinner("Generating audio..."):
                        audio_bytes = generate_tts_audio(content)
                        if audio_bytes:
                            st.session_state[tts_audio_key] = audio_bytes
                            st.session_state[tts_generate_key] = False
                            st.session_state[tts_autoplay_key] = True
                            st.rerun()
                        else:
                            st.error("Could not generate audio. Please check your API key and connection.")
                            st.session_state[tts_generate_key] = False
                
                # Display audio player if audio is available
                if st.session_state.get(tts_audio_key) and isinstance(st.session_state[tts_audio_key], bytes):
                    autoplay = st.session_state.get(tts_autoplay_key, False)
                    st.audio(st.session_state[tts_audio_key], format="audio/mp3", autoplay=autoplay)
                    # Reset autoplay after first play
                    if autoplay:
                        st.session_state[tts_autoplay_key] = False


def display_vocabulary_metrics():
    """Show simple vocabulary metrics in the sidebar."""
    st.sidebar.header("Vocabulary Metrics")
    st.sidebar.metric("Active Vocabulary", len(st.session_state["user_vocabulary"]))
    st.sidebar.metric("Passive Vocabulary", len(st.session_state["ai_vocabulary"]))


@st.cache_data(show_spinner="Processing analytics...")
def process_analytics_data(saved_chats, session_analysis_entries):
    """
    Process saved chats and current session analytics into DataFrames and metrics.
    Cached to avoid re-parsing JSON history on every render.
    """
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
        except ValueError:
            # We might log this, but inside a cached function st.warning might duplicate
            continue

        parsed_chats.append(
            {
                "name": chat["name"],
                "saved_at": saved_at,
                "raw_saved_at": raw_saved_at,
                "messages": payload.get("messages", []),
                "analytics": payload.get("analytics", []),
            }
        )

    saved_has_analytics = any(chat["analytics"] for chat in parsed_chats)
    
    # Only add current session if it has analytics that aren't just repeats of saved ones
    # or if it's a strictly new set of analytics.
    if session_analysis_entries and not saved_has_analytics:
        # Note: In a real app we might want better de-duplication logic, 
        # but here we assume session is distinct if nothing is saved or if we just want to see it.
        # A simple heuristic: if the user is viewing analytics, they probably want to see the current session too.
        # The original logic was: "if session_analysis_entries and not saved_has_analytics".
        # We'll stick to the original logic to preserve behavior.
        parsed_chats.append(
            {
                "name": "Current Session (unsaved)",
                "saved_at": datetime.datetime.utcnow(),
                "raw_saved_at": None,
                "messages": [],  # Messages not strictly needed for analytics aggregation
                "analytics": session_analysis_entries,
                "is_session": True,
            }
        )

    if not parsed_chats:
        return None

    parsed_chats.sort(
        key=lambda item: (item["saved_at"] is None, item["saved_at"] or datetime.datetime.min)
    )

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
    
    # New metrics
    total_sentences = 0
    error_free_sentences = 0
    error_timeline_records = []
    recent_errors = []

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

            # Ensure overall_error_types is a list of strings
            if overall_error_types and isinstance(overall_error_types[0], str):
                errors_str = ", ".join(overall_error_types)
            else:
                errors_str = "none"

            # Ensure overall_topics is a list of strings
            if overall_topics and isinstance(overall_topics[0], str):
                topics_str = ", ".join(overall_topics)
            else:
                topics_str = "none"

            message_records.append(
                {
                    "Conversation": chat["name"],
                    "Conversation #": idx,
                    "Saved At": chat["saved_at"],
                    "Message Timestamp": message_dt,
                    "Summary": analysis_data.get("message_summary", ""),
                    "Overall Errors": errors_str,
                    "Overall Topics": topics_str,
                    "Source": analysis_data.get("source", "unknown"),
                }
            )

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
                    tense_timeline_records.append(
                        {
                            "Conversation #": idx,
                            "Chat": chat["name"],
                            "Saved At": chat["saved_at"],
                            "Message Timestamp": message_dt,
                            "Tense": normalized_tense,
                            "Count": 1,
                        }
                    )

                for err in error_types:
                    normalized_error = (err or "").strip().lower()
                    if normalized_error == "none":
                        continue
                    # Exclude punctuation from aggregate metrics as per user request
                    if normalized_error == "punctuation":
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
                
                # Proficiency & Error Tracking
                total_sentences += 1
                has_error = False
                current_sentence_errors = []
                
                for err in error_types:
                    normalized_err = (err or "").strip().lower()
                    if normalized_err != "none" and normalized_err != "punctuation":
                        has_error = True
                        current_sentence_errors.append(normalized_err)
                
                if not has_error:
                    error_free_sentences += 1
                else:
                    # Add to recent errors
                    recent_errors.append({
                        "Saved At": chat["saved_at"],
                        "Message Timestamp": message_dt,
                        "Sentence": sentence.get("sentence_text", ""),
                        "Error Types": ", ".join(current_sentence_errors),
                        "Feedback": sentence.get("feedback", ""),
                        "Conversation": chat["name"]
                    })
                
                # Add to timeline (one record per sentence to allow aggregation)
                error_timeline_records.append({
                    "Saved At": chat["saved_at"],
                    "Message Timestamp": message_dt,
                    "Conversation #": idx,
                    "Conversation": chat["name"],
                    "Has Error": has_error,
                    "Error Count": len(current_sentence_errors)
                })

        if analyzed_messages == 0:
            chats_missing_analysis.append(chat["name"])

        dominant_tense = conversation_tenses.most_common(1)
        dominant_error = conversation_errors.most_common(1)
        top_topics = [label for label, _ in conversation_topics.most_common(3)]

        new_words = conversation_vocab - global_vocab
        global_vocab.update(conversation_vocab)

        growth_records.append(
            {
                "Saved At": chat["saved_at"],
                "Conversation #": idx,
                "Cumulative Unique Words": len(global_vocab),
            }
        )
        new_words_records.append(
            {
                "Saved At": chat["saved_at"],
                "Conversation #": idx,
                "New Words": len(new_words),
            }
        )

        processed_chats.append(
            {
                "Chat": chat["name"],
                "Saved At": chat["saved_at"],
                "Conversation #": idx,
                "Analyzed Messages": analyzed_messages,
                "Unique Vocabulary": len(conversation_vocab),
                "Dominant Tense": dominant_tense[0][0] if dominant_tense else "N/A",
                "Top Error Type": dominant_error[0][0] if dominant_error else "N/A",
                "Primary Topics": ", ".join(top_topics) if top_topics else "N/A",
                "Latest Takeaway": takeaways[-1] if takeaways else "",
                "Source": "Mixta"
                if any(
                    (entry.get("analysis") or {}).get("source") != "gpt"
                    for entry in analytics_entries
                )
                else "GPT",
            }
        )

    if not analysis_available:
        return {"analysis_available": False}

    summary_df = pd.DataFrame(processed_chats)
    summary_df["Saved At Display"] = summary_df["Saved At"].apply(
        lambda dt: dt.strftime("%Y-%m-%d %H:%M")
        if isinstance(dt, datetime.datetime) and pd.notna(dt)
        else "Not recorded"
    )

    growth_df = pd.DataFrame(growth_records)
    new_words_df = pd.DataFrame(new_words_records)
    tense_overall_df = (
        pd.DataFrame(
            [{"Tense": tense, "Count": count} for tense, count in tense_counter.items()]
        ).sort_values("Count", ascending=False)
        if tense_counter
        else pd.DataFrame()
    )
    error_overall_df = (
        pd.DataFrame(
            [{"Error Type": error, "Count": count} for error, count in error_counter.items()]
        ).sort_values("Count", ascending=False)
        if error_counter
        else pd.DataFrame()
    )
    topic_overall_df = (
        pd.DataFrame(
            [{"Topic": topic, "Count": count} for topic, count in topic_counter.items()]
        ).sort_values("Count", ascending=False)
        if topic_counter
        else pd.DataFrame()
    )
    vocab_category_df = (
        pd.DataFrame(
            [
                {"Category": category, "Count": count}
                for category, count in vocab_category_counter.items()
            ]
        ).sort_values("Count", ascending=False)
        if vocab_category_counter
        else pd.DataFrame()
    )
    top_words_df = (
        pd.DataFrame(
            [{"Word": word, "Frequency": freq} for word, freq in vocab_counter.most_common(20)]
        )
        if vocab_counter
        else pd.DataFrame()
    )
    tense_timeline_df = pd.DataFrame(tense_timeline_records)
    message_detail_df = pd.DataFrame(message_records)

    top_tense = tense_counter.most_common(1)
    top_error = error_counter.most_common(1)
    top_topic = topic_counter.most_common(1)
    
    # Finalize new metrics
    proficiency_score = (error_free_sentences / total_sentences * 100) if total_sentences > 0 else 0
    error_timeline_df = pd.DataFrame(error_timeline_records)
    recent_errors_df = pd.DataFrame(recent_errors)
    
    # Sort recent errors by time descending
    if not recent_errors_df.empty:
        # Sort by Saved At first (as proxy for time if msg timestamp missing)
        recent_errors_df.sort_values(by=["Saved At", "Message Timestamp"], ascending=False, inplace=True)

    return {
        "analysis_available": True,
        "summary_df": summary_df,
        "growth_df": growth_df,
        "new_words_df": new_words_df,
        "tense_overall_df": tense_overall_df,
        "error_overall_df": error_overall_df,
        "topic_overall_df": topic_overall_df,
        "vocab_category_df": vocab_category_df,
        "top_words_df": top_words_df,
        "tense_timeline_df": tense_timeline_df,
        "message_detail_df": message_detail_df,
        "global_vocab": sorted(global_vocab),
        "chats_missing_analysis": chats_missing_analysis,
        "total_analyzed_messages": int(summary_df["Analyzed Messages"].sum()),
        "top_tense": top_tense,
        "top_error": top_error,
        "top_topic": top_topic,
        "proficiency_score": proficiency_score,
        "error_timeline_df": error_timeline_df,
        "recent_errors_df": recent_errors_df,
    }


def analytics_dashboard():
    """Render the analytics dashboard from stored GPT evaluations."""
    st.title("Learning Analytics Dashboard")
    st.caption("Insights generated from GPT-5.1-mini sentence-level evaluations.")

    with st.sidebar:
        if st.button("Back to Chat"):
            st.session_state["show_analytics"] = False
            st.rerun()

    saved_chats = load_saved_chats(include_history=True)
    session_analysis_entries = [
        entry
        for entry in st.session_state.get("message_analytics", [])
        if entry.get("analysis")
    ]

    if not saved_chats and not session_analysis_entries:
        st.info("Save at least one conversation to see your analytics.")
        return

    # Use cached processing
    results = process_analytics_data(saved_chats, session_analysis_entries)

    if not results:
        st.info(
            "No analyzable conversations found. Try saving a new chat or send a new "
            "message to generate analytics."
        )
        return

    if not results.get("analysis_available"):
        st.info("No GPT-powered analytics found yet. Send new messages to generate insights.")
        return

    # Unpack results
    summary_df = results["summary_df"]
    growth_df = results["growth_df"]
    new_words_df = results["new_words_df"]
    tense_overall_df = results["tense_overall_df"]
    error_overall_df = results["error_overall_df"]
    topic_overall_df = results["topic_overall_df"]
    vocab_category_df = results["vocab_category_df"]
    top_words_df = results["top_words_df"]
    tense_timeline_df = results["tense_timeline_df"]
    message_detail_df = results["message_detail_df"]
    global_vocab = results["global_vocab"]
    chats_missing_analysis = results["chats_missing_analysis"]
    total_analyzed_messages = results["total_analyzed_messages"]
    top_tense = results["top_tense"]
    top_error = results["top_error"]
    top_topic = results["top_topic"]
    proficiency_score = results.get("proficiency_score", 0)
    error_timeline_df = results.get("error_timeline_df", pd.DataFrame())
    recent_errors_df = results.get("recent_errors_df", pd.DataFrame())
    
    # === FILTERS SECTION ===
    st.markdown("### Filters")
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        # Conversation selector
        conversation_names = ["All Conversations"] + summary_df["Chat"].tolist()
        selected_conversations = st.multiselect(
            "Select Conversations",
            conversation_names,
            default=["All Conversations"]
        )
    
    with col_filter2:
        # Date range filter
        if not summary_df["Saved At"].dropna().empty:
            min_date = summary_df["Saved At"].min().date() if pd.notna(summary_df["Saved At"].min()) else None
            max_date = summary_df["Saved At"].max().date() if pd.notna(summary_df["Saved At"].max()) else None
            
            if min_date and max_date:
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
        else:
            date_range = None
    
    # Apply filters
    filtered_summary_df = summary_df.copy()
    if "All Conversations" not in selected_conversations and selected_conversations:
        filtered_summary_df = summary_df[summary_df["Chat"].isin(selected_conversations)]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_summary_df = filtered_summary_df[
            (filtered_summary_df["Saved At"].dt.date >= start_date) &
            (filtered_summary_df["Saved At"].dt.date <= end_date)
        ]
    
    # Filter related dataframes
    filtered_conversation_nums = filtered_summary_df["Conversation #"].tolist()
    filtered_growth_df = growth_df[growth_df["Conversation #"].isin(filtered_conversation_nums)]
    filtered_new_words_df = new_words_df[new_words_df["Conversation #"].isin(filtered_conversation_nums)]
    
    st.markdown("---")
    
    # === ENHANCED METRICS WITH DELTAS ===
    fallback_rows = filtered_summary_df[filtered_summary_df["Source"] != "GPT"]
    
    # Calculate metrics for current and previous period
    total_conversations = len(filtered_summary_df)
    total_messages = filtered_summary_df["Analyzed Messages"].sum()
    
    # Calculate vocabulary delta (comparing last half vs first half)
    if len(filtered_growth_df) >= 2:
        midpoint = len(filtered_growth_df) // 2
        early_vocab = filtered_growth_df.iloc[:midpoint]["Cumulative Unique Words"].iloc[-1] if midpoint > 0 else 0
        recent_vocab = filtered_growth_df.iloc[midpoint:]["Cumulative Unique Words"].iloc[-1] if len(filtered_growth_df) > midpoint else 0
        vocab_delta = recent_vocab - early_vocab
    else:
        vocab_delta = 0
        recent_vocab = filtered_growth_df["Cumulative Unique Words"].iloc[-1] if not filtered_growth_df.empty else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Conversations", total_conversations)
    col2.metric("Analyzed Messages", int(total_messages))
    col3.metric("Unique Vocabulary", int(recent_vocab), delta=f"{vocab_delta:+d}" if vocab_delta != 0 else None)

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Most Used Tense",
        f"{top_tense[0][0]} ({top_tense[0][1]})" if top_tense else "N/A",
    )
    col5.metric(
        "Top Error Type",
        f"{top_error[0][0]} ({top_error[0][1]})" if top_error else "N/A",
    )
    col6.metric(
        "Top Topic",
        f"{top_topic[0][0]} ({top_topic[0][1]})" if top_topic else "N/A",
    )
    
    # Proficiency Metric (New)
    st.markdown("---")
    col_prof1, col_prof2 = st.columns([1, 3])
    with col_prof1:
        st.metric("Grammar Proficiency", f"{proficiency_score:.1f}%", help="Percentage of sentences without errors")
    with col_prof2:
        # Simple progress bar for proficiency
        st.caption("Accuracy Rate")
        st.progress(min(proficiency_score / 100.0, 1.0))

    if not fallback_rows.empty:
        st.info(
            "Algunas conversaciones usan análisis heurístico porque la llamada a GPT falló o no está disponible."
        )

    st.markdown("---")
    
    # === TABS SECTION ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Vocabulary", "Grammar", "Topics", "Flashcards"])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.subheader("Conversation Summary")
        display_df = filtered_summary_df[
            [
                "Chat",
                "Saved At Display",
                "Source",
                "Analyzed Messages",
                "Unique Vocabulary",
                "Dominant Tense",
                "Top Error Type",
                "Primary Topics",
                "Latest Takeaway",
            ]
        ].rename(columns={"Saved At Display": "Saved At"})
        st.dataframe(display_df, use_container_width=True)

        if not message_detail_df.empty:
            with st.expander("Message-Level Details"):
                detail_view = message_detail_df.copy()
                detail_view["Saved At"] = detail_view["Saved At"].apply(
                    lambda dt: dt.strftime("%Y-%m-%d %H:%M")
                    if isinstance(dt, datetime.datetime) and pd.notna(dt)
                    else "Not recorded"
                )
                detail_view["Message Timestamp"] = detail_view["Message Timestamp"].apply(
                    lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(dt, datetime.datetime) and pd.notna(dt)
                    else "Not recorded"
                )
                st.dataframe(detail_view, use_container_width=True)
    
    # TAB 2: VOCABULARY
    with tab2:
        has_timestamp = not filtered_growth_df["Saved At"].dropna().empty
        
        if not filtered_growth_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Vocabulary Growth")
            if has_timestamp:
                growth_chart_df = filtered_growth_df.dropna(subset=["Saved At"]).copy()
                growth_chart_df["Saved At"] = pd.to_datetime(growth_chart_df["Saved At"])
                growth_chart_df["Saved Date"] = growth_chart_df["Saved At"].dt.normalize()
                daily_growth_df = (
                    growth_chart_df.groupby("Saved Date", as_index=False)["Cumulative Unique Words"]
                    .max()
                )
                growth_chart = (
                    alt.Chart(daily_growth_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Saved Date:T", title="Date"),
                        y=alt.Y("Cumulative Unique Words:Q", title="Cumulative Unique Words"),
                        tooltip=["Saved Date:T", "Cumulative Unique Words:Q"],
                    )
                    .configure_axis(
                        gridColor="rgba(255, 255, 255, 0.1)",
                        domainColor="rgba(255, 255, 255, 0.1)",
                        labelColor="#94a3b8",
                        titleColor="#94a3b8"
                    )
                    .configure_view(strokeWidth=0)
                    .properties(height=300, background="transparent")
                )
            else:
                growth_chart = (
                    alt.Chart(filtered_growth_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Conversation #:Q", title="Conversation"),
                        y=alt.Y("Cumulative Unique Words:Q", title="Cumulative Unique Words"),
                        tooltip=["Conversation #:Q", "Cumulative Unique Words:Q"],
                    )
                    .configure_axis(
                        gridColor="rgba(255, 255, 255, 0.1)",
                        domainColor="rgba(255, 255, 255, 0.1)",
                        labelColor="#94a3b8",
                        titleColor="#94a3b8"
                    )
                    .configure_view(strokeWidth=0)
                    .properties(height=300, background="transparent")
                )
            st.altair_chart(growth_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if not filtered_new_words_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("New Vocabulary Introduced")
            if has_timestamp:
                new_words_chart_df = filtered_new_words_df.dropna(subset=["Saved At"]).copy()
                new_words_chart_df["Saved At"] = pd.to_datetime(new_words_chart_df["Saved At"])
                new_words_chart_df["Saved Date"] = new_words_chart_df["Saved At"].dt.normalize()
                daily_new_words_df = (
                    new_words_chart_df.groupby("Saved Date", as_index=False)["New Words"].sum()
                )
                new_words_chart = (
                    alt.Chart(daily_new_words_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Saved Date:T", title="Date"),
                        y=alt.Y("New Words:Q", title="New Words Introduced"),
                        tooltip=["Saved Date:T", "New Words:Q"],
                    )
                    .properties(height=250)
                )
            else:
                new_words_chart = (
                    alt.Chart(filtered_new_words_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Conversation #:Q", title="Conversation"),
                        y=alt.Y("New Words:Q", title="New Words Introduced"),
                        tooltip=["Conversation #:Q", "New Words:Q"],
                    )
                    .properties(height=250)
                )
            st.altair_chart(new_words_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not vocab_category_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Vocabulary Focus Areas")
            vocab_category_chart = (
                alt.Chart(vocab_category_df)
                .mark_bar(color="#FFC914")
                .encode(
                    x=alt.X("Count:Q", title="Occurrences"),
                    y=alt.Y("Category:N", sort="-x", title="Category"),
                    tooltip=["Category:N", "Count:Q"],
                )
                .properties(height=240)
            )
            st.altair_chart(vocab_category_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if not top_words_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Top Vocabulary (by frequency)")
            top_words_chart = (
                alt.Chart(top_words_df)
                .mark_bar()
                .encode(
                    x=alt.X("Frequency:Q", title="Frequency"),
                    y=alt.Y("Word:N", sort="-x", title="Word"),
                    tooltip=["Word:N", "Frequency:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(top_words_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if global_vocab:
            vocab_df = pd.DataFrame({"Word": global_vocab})
            csv_bytes = vocab_df.to_csv(index=False)
            st.download_button(
                "Download Vocabulary as CSV",
                csv_bytes,
                "palabrero_vocabulary.csv",
                "text/csv",
            )
    
    # TAB 3: GRAMMAR
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Tense Usage")
            if not tense_overall_df.empty:
                tense_chart = (
                    alt.Chart(tense_overall_df)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta=alt.Theta("Count", stack=True),
                        color=alt.Color("Tense", legend=None),
                        tooltip=["Tense", "Count"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(tense_chart, use_container_width=True)
                st.dataframe(tense_overall_df, use_container_width=True, hide_index=True)
            else:
                st.info("No tense data available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Error Distribution")
            if not error_overall_df.empty:
                error_chart = (
                    alt.Chart(error_overall_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count", title="Count"),
                        y=alt.Y("Error Type", sort="-x", title="Error Type"),
                        color=alt.Color("Error Type", legend=None),
                        tooltip=["Error Type", "Count"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(error_chart, use_container_width=True)
                st.dataframe(error_overall_df, use_container_width=True, hide_index=True)
            else:
                st.info("No error data available.")
            st.markdown('</div>', unsafe_allow_html=True)

        if not tense_timeline_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Tense Usage Over Time")
            timeline_chart = (
                alt.Chart(tense_timeline_df)
                .mark_circle()
                .encode(
                    x=alt.X("Conversation #:Q", title="Conversation"),
                    y=alt.Y("Tense", title="Tense"),
                    size="count()",
                    color="Tense",
                    tooltip=["Conversation #", "Tense", "count()"],
                )
                .properties(height=300)
            )
            st.altair_chart(timeline_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Error Timeline
        if not error_timeline_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Error Frequency Over Time")
            error_timeline_chart = (
                alt.Chart(error_timeline_df)
                .mark_bar()
                .encode(
                    x=alt.X("Conversation #:Q", title="Conversation"),
                    y=alt.Y("Error Count:Q", title="Errors per Sentence"),
                    color=alt.Color("Has Error:N", scale={"domain": [True, False], "range": ["#f43f5e", "#10b981"]}),
                    tooltip=["Conversation", "Error Count", "Has Error"]
                )
                .properties(height=300)
            )
            st.altair_chart(error_timeline_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


        # Detailed Feedback Table
        if not recent_errors_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Recent Areas for Improvement")
            
            # Format for display
            display_errors = recent_errors_df[["Sentence", "Error Types", "Feedback", "Conversation"]].head(10)
            st.dataframe(
                display_errors,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Sentence": st.column_config.TextColumn("Sentence", width="medium"),
                    "Error Types": st.column_config.TextColumn("Error Type", width="small"),
                    "Feedback": st.column_config.TextColumn("Feedback", width="large"),
                    "Conversation": st.column_config.TextColumn("Chat", width="small"),
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: TOPICS
    with tab4:
        if not topic_overall_df.empty:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Topic Coverage")
            topic_chart = (
                alt.Chart(topic_overall_df)
                .mark_bar(color="#17BEBB")
                .encode(
                    x=alt.X("Count:Q", title="Mentions"),
                    y=alt.Y("Topic:N", sort="-x", title="Topic"),
                    tooltip=["Topic:N", "Count:Q"],
                )
                .properties(height=280)
            )
            st.altair_chart(topic_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 5: FLASHCARDS (MOCHI)
    with tab5:
        st.subheader("Mochi Flashcards Generator")
        st.markdown("""
        Generate flashcards from your mistakes in selected conversations. 
        Download the `.mochi` file and import it into [Mochi](https://mochi.cards/).
        """)
        
        # Use the filtered summary to get the relevant chats
        selected_chat_names = filtered_summary_df["Chat"].unique().tolist()
        
        # Initialize session state for cards if not present
        if "generated_flashcards" not in st.session_state:
            st.session_state["generated_flashcards"] = []

        if st.button("Generate Flashcards from Filtered View"):
            target_chats = []
            
            # Check saved chats
            for chat in saved_chats:
                if chat["name"] in selected_chat_names:
                    try:
                        payload = parse_chat_payload(chat.get("history"))
                        target_chats.append({
                            "name": chat["name"],
                            "analytics": payload.get("analytics", [])
                        })
                    except:
                        pass
            
            # Check current session
            if "Current Session (unsaved)" in selected_chat_names:
                 target_chats.append({
                    "name": "Current Session",
                    "analytics": session_analysis_entries
                })
            
            if not target_chats:
                st.warning("No conversations found to generate cards from.")
            else:
                # Generate 10 cards max by default
                cards = generate_cloze_cards(target_chats, limit=10)
                
                if not cards:
                    st.info("No mistakes found suitable for flashcards in these conversations.")
                else:
                    st.session_state["generated_flashcards"] = cards
                    st.success(f"Generated {len(cards)} flashcards!")
        
        # Display and Manage Cards
        if st.session_state["generated_flashcards"]:
            cards = st.session_state["generated_flashcards"]
            st.write(f"**Previewing {len(cards)} cards**")
            
            # List cards with delete buttons
            cards_to_keep = []
            for i, card in enumerate(cards):
                with st.expander(f"Card {i+1}: {card['name']}", expanded=False):
                    st.text_area("Content", card["content"], height=150, key=f"card_preview_{i}", disabled=True)
                    if not st.checkbox("Delete this card", key=f"delete_card_{i}"):
                        cards_to_keep.append(card)
            
            # Update state if deletions occurred (this logic is a bit tricky in Streamlit immediate mode)
            # Better approach: "Update Deck" button or just generate download from "cards_to_keep"
            # But "cards_to_keep" is rebuilt every run based on checkboxes.
            
            if len(cards_to_keep) < len(cards):
                st.warning(f"You have marked {len(cards) - len(cards_to_keep)} cards for deletion. They will be excluded from the download.")

            # Download
            if cards_to_keep:
                mochi_zip = create_mochi_zip("Palabrero Mistakes", cards_to_keep)
                st.download_button(
                    label="Download .mochi Deck (ZIP)",
                    data=mochi_zip,
                    file_name="palabrero_flashcards.mochi",
                    mime="application/zip"
                )
            else:
                st.error("No cards remaining to download.")

    if chats_missing_analysis:
        st.info(
            "These chats do not yet have GPT analytics and were skipped: "
            + ", ".join(chats_missing_analysis)
        )


def export_vocabulary():
    """Allow user to export current session vocabulary as CSV."""
    vocab_df = pd.DataFrame({"Word": list(st.session_state["user_vocabulary"])})
    csv = vocab_df.to_csv(index=False)
    st.download_button(
        "Download Vocabulary as CSV", csv, "vocabulary.csv", "text/csv"
    )



