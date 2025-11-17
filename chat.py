import streamlit as st

from analysis import analyze_user_message, record_message_analysis
from vocabulary import update_vocabulary


def chat_handling(user_input: str) -> str:
    """Handle the chat flow: send user input + system message to the model."""
    with st.spinner("Generating AI response..."):
        client = st.session_state["openai_client"]

        system_message = {
            "role": "system",
            "content": st.session_state["system_prompt"],
        }

        messages = [system_message] + st.session_state["chat_history"] + [
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=150,
        )

    ai_message = response.choices[0].message.content

    # Update chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Update vocabulary and analytics
    update_vocabulary(user_input, ai_message)
    analysis_result = analyze_user_message(user_input)
    record_message_analysis(user_input, analysis_result)

    return ai_message


