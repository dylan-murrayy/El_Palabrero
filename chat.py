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


def chat_handling(user_input: str) -> str:
    """Handle the chat flow: send user input + system message to the model."""
    with st.spinner("Generating AI response..."):
        client = st.session_state.get("openai_client")
        if client is None:
            st.error("OpenAI client is not initialized. Please check your API key.")
            ai_message = "Lo siento, hubo un problema al inicializar el cliente de OpenAI."
        else:
            system_message = {
                "role": "system",
                "content": st.session_state.get("system_prompt", ""),
            }

            messages = [system_message] + st.session_state["chat_history"] + [
                {"role": "user", "content": user_input}
            ]

            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    max_completion_tokens=150,
                )
                # Try to extract a normal-text AI message first
                ai_message = _extract_ai_message(response)

                # If we still have nothing, fall back to a debug dump so we can
                # see what the API actually returned.
                if not ai_message:
                    raw_msg = None
                    try:
                        if getattr(response, "choices", None):
                            raw_msg = response.choices[0].message
                        else:
                            raw_msg = response
                    except Exception:  # noqa: BLE001
                        raw_msg = response

                    ai_message = (
                        "[DEBUG] OpenAI devolvió una respuesta sin texto útil.\n\n"
                        f"Objeto crudo del primer mensaje:\n{raw_msg!r}"
                    )
            except Exception as err:  # noqa: BLE001
                st.error(f"Error al llamar a la API de OpenAI: {err}")
                ai_message = (
                    "Lo siento, hubo un error al contactar con la API de OpenAI."
                )

    # Update chat history (always show *something* to avoid blank AI messages)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Update vocabulary and analytics
    update_vocabulary(user_input, ai_message)
    analysis_result = analyze_user_message(user_input)
    record_message_analysis(user_input, analysis_result)

    return ai_message


