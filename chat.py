import streamlit as st

from analysis import ANALYSIS_MODEL, analyze_user_message, record_message_analysis
from vocabulary import update_vocabulary


# Use a stable chat model that works well with the chat.completions API.
CHAT_MODEL = ANALYSIS_MODEL


def _extract_ai_message(response) -> str:
    """
    Safely extract the assistant's message text from an OpenAI chat completion
    response, handling possible structural differences or empty outputs.
    """
    if not response or not getattr(response, "choices", None):
        return ""

    choice = response.choices[0]
    message = getattr(choice, "message", None)

    # Common path: .message.content is already a string
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()

    # Some SDK/response variants may use list-of-parts content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Typical pattern: {"type": "output_text", "text": {"value": "..."}} or similar
                text_obj = part.get("text") or part.get("value") or part.get("content")
                if isinstance(text_obj, str):
                    parts.append(text_obj)
                elif isinstance(text_obj, dict):
                    val = (
                        text_obj.get("value")
                        or text_obj.get("content")
                        or text_obj.get("text")
                    )
                    if isinstance(val, str):
                        parts.append(val)
        return " ".join(parts).strip()

    # If for some reason content is dict-like
    if isinstance(content, dict):
        for key in ("content", "value", "text"):
            val = content.get(key)
            if isinstance(val, str):
                return val.strip()

    # Fallback: try dict access on message itself
    if isinstance(message, dict):
        val = message.get("content")
        if isinstance(val, str):
            return val.strip()

    return ""


from typing import Iterator

def get_chat_stream(user_input: str) -> Iterator[str]:
    """
    Generator that yields chunks of the AI response from OpenAI.
    """
    client = st.session_state.get("openai_client")
    if client is None:
        yield "Lo siento, hubo un problema al inicializar el cliente de OpenAI."
        return

    system_message = {
        "role": "system",
        "content": st.session_state.get("system_prompt", ""),
    }

    messages = [system_message] + st.session_state["chat_history"] + [
        {"role": "user", "content": user_input}
    ]

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_completion_tokens=300,
            stream=True,
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as err:  # noqa: BLE001
        yield f"Error al llamar a la API de OpenAI: {err}"


def process_chat_turn(user_input: str, ai_response: str):
    """
    Handle post-processing after the full response has been received:
    - Update vocabulary
    - Run analytics
    """
    # Update vocabulary and analytics
    update_vocabulary(user_input, ai_response)
    analysis_result = analyze_user_message(user_input)
    record_message_analysis(user_input, analysis_result)
