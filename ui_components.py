import datetime
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from analysis import (
    ALLOWED_ERROR_TYPES,
    ALLOWED_TENSES,
    ALLOWED_TOPICS,
    ALLOWED_VOCAB_CATEGORIES,
)
from storage import load_saved_chats, parse_chat_payload


def display_chat():
    """Render the main chat conversation."""
    st.title("Palabrero - Aprende Español")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")


def display_vocabulary_metrics():
    """Show simple vocabulary metrics in the sidebar."""
    st.sidebar.header("Vocabulary Metrics")
    st.sidebar.metric("Active Vocabulary", len(st.session_state["user_vocabulary"]))
    st.sidebar.metric("Passive Vocabulary", len(st.session_state["ai_vocabulary"]))


def analytics_dashboard():
    """Render the analytics dashboard from stored GPT evaluations."""
    st.title("Learning Analytics Dashboard")
    st.caption("Insights generated from GPT-5.1-mini sentence-level evaluations.")

    with st.sidebar:
        if st.button("Back to Chat"):
            st.session_state["show_analytics"] = False
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

        parsed_chats.append(
            {
                "name": chat["name"],
                "saved_at": saved_at,
                "raw_saved_at": raw_saved_at,
                "messages": payload.get("messages", []),
                "analytics": payload.get("analytics", []),
            }
        )

    session_analysis_entries = [
        entry
        for entry in st.session_state.get("message_analytics", [])
        if entry.get("analysis")
    ]
    saved_has_analytics = any(chat["analytics"] for chat in parsed_chats)
    if session_analysis_entries and not saved_has_analytics:
        parsed_chats.append(
            {
                "name": "Current Session (unsaved)",
                "saved_at": datetime.datetime.utcnow(),
                "raw_saved_at": None,
                "messages": st.session_state.get("chat_history", []),
                "analytics": session_analysis_entries,
                "is_session": True,
            }
        )

    if not parsed_chats:
        st.info(
            "No analyzable conversations found. Try saving a new chat or send a new "
            "message to generate analytics."
        )
        return

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

            message_records.append(
                {
                    "Conversation": chat["name"],
                    "Conversation #": idx,
                    "Saved At": chat["saved_at"],
                    "Message Timestamp": message_dt,
                    "Summary": analysis_data.get("message_summary", ""),
                    "Overall Errors": ", ".join(overall_error_types)
                    if overall_error_types
                    else "none",
                    "Overall Topics": ", ".join(overall_topics)
                    if overall_topics
                    else "none",
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
        st.info("No GPT-powered analytics found yet. Send new messages to generate insights.")
        return

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

    total_analyzed_messages = int(summary_df["Analyzed Messages"].sum())
    fallback_rows = summary_df[summary_df["Source"] != "GPT"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Conversations", len(summary_df))
    col2.metric("Analyzed Messages", total_analyzed_messages)
    col3.metric("Unique Vocabulary (GPT)", len(global_vocab))

    top_tense = tense_counter.most_common(1)
    top_error = error_counter.most_common(1)
    top_topic = topic_counter.most_common(1)

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

    if not fallback_rows.empty:
        st.info(
            "Algunas conversaciones usan análisis heurístico porque la llamada a GPT falló o no está disponible."
        )

    st.markdown("---")

    has_timestamp = not growth_df["Saved At"].dropna().empty

    if not growth_df.empty:
        if has_timestamp:
            growth_chart_df = growth_df.dropna(subset=["Saved At"]).copy()
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
                .properties(height=300)
            )
        else:
            growth_chart = (
                alt.Chart(growth_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Conversation #:Q", title="Conversation"),
                    y=alt.Y("Cumulative Unique Words:Q", title="Cumulative Unique Words"),
                    tooltip=["Conversation #:Q", "Cumulative Unique Words:Q"],
                )
                .properties(height=300)
            )
        st.subheader("Vocabulary Growth")
        st.altair_chart(growth_chart, use_container_width=True)

    if not new_words_df.empty:
        if has_timestamp:
            new_words_chart_df = new_words_df.dropna(subset=["Saved At"]).copy()
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
                alt.Chart(new_words_df)
                .mark_bar()
                .encode(
                    x=alt.X("Conversation #:Q", title="Conversation"),
                    y=alt.Y("New Words:Q", title="New Words Introduced"),
                    tooltip=["Conversation #:Q", "New Words:Q"],
                )
                .properties(height=250)
            )
        st.subheader("New Vocabulary Introduced")
        st.altair_chart(new_words_chart, use_container_width=True)

    if not tense_overall_df.empty:
        st.subheader("Verb Tense Usage")
        tense_chart = (
            alt.Chart(tense_overall_df)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Occurrences"),
                y=alt.Y("Tense:N", sort="-x", title="Verb Tense"),
                tooltip=["Tense:N", "Count:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(tense_chart, use_container_width=True)

        if not tense_timeline_df.empty:
            if tense_timeline_df["Message Timestamp"].notna().any():
                timeline_df = tense_timeline_df.dropna(subset=["Message Timestamp"]).copy()
                timeline_df["Message Timestamp"] = pd.to_datetime(
                    timeline_df["Message Timestamp"]
                )
                timeline_chart = (
                    alt.Chart(timeline_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Message Timestamp:T", title="Message Timestamp"),
                        y=alt.Y(
                            "Count:Q",
                            aggregate="sum",
                            title="Occurrences",
                        ),
                        color=alt.Color("Tense:N", title="Verb Tense"),
                        tooltip=["Message Timestamp:T", "Tense:N", "Count:Q"],
                    )
                    .properties(height=320)
                )
            else:
                timeline_chart = (
                    alt.Chart(tense_timeline_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Conversation #:Q", title="Conversation"),
                        y=alt.Y("Count:Q", aggregate="sum", title="Occurrences"),
                        color=alt.Color("Tense:N", title="Verb Tense"),
                        tooltip=["Conversation #:Q", "Tense:N", "Count:Q"],
                    )
                    .properties(height=320)
                )
            st.altair_chart(timeline_chart, use_container_width=True)

    if not error_overall_df.empty:
        st.subheader("Error Type Distribution")
        error_chart = (
            alt.Chart(error_overall_df)
            .mark_bar(color="#E4572E")
            .encode(
                x=alt.X("Count:Q", title="Occurrences"),
                y=alt.Y("Error Type:N", sort="-x", title="Error Type"),
                tooltip=["Error Type:N", "Count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(error_chart, use_container_width=True)

    if not topic_overall_df.empty:
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

    if not vocab_category_df.empty:
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

    if not top_words_df.empty:
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

    st.subheader("Conversation Summary")
    display_df = summary_df[
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
        st.subheader("Message-Level Details")
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

    st.markdown("---")
    aggregated_vocab = sorted(global_vocab)
    if aggregated_vocab:
        vocab_df = pd.DataFrame({"Word": aggregated_vocab})
        csv_bytes = vocab_df.to_csv(index=False)
        st.subheader("Download Your GPT-Derived Vocabulary")
        st.download_button(
            "Download Vocabulary as CSV",
            csv_bytes,
            "palabrero_vocabulary.csv",
            "text/csv",
        )

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


