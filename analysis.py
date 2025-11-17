import datetime
import json
import re
import streamlit as st

from vocabulary import extract_words


# Use a stable chat model that works with the chat.completions API.
ANALYSIS_MODEL = "gpt-4.1-mini"


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
    "other",
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
    "other",
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
    "other",
]

ALLOWED_VOCAB_CATEGORIES = [
    "new-word",
    "advanced-word",
    "review-word",
    "idiom",
    "collocation",
]


def analyze_user_message(user_text: str):
    """Call OpenAI chat completions and parse JSON analysis."""
    client = st.session_state.get("openai_client")
    if not user_text.strip():
        return None
    if client is None:
        st.info("Analítica GPT no disponible. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="no_client")

    system_message = (
        "Eres Palabrero, un tutor de español.\n"
        "TAREA: Analiza el mensaje del estudiante y devuelve SOLO un JSON válido (sin texto adicional), "
        "con este esquema exacto:\n\n"
        "{\n"
        '  \"message_summary\": \"resumen breve en español\",\n'
        '  \"sentences\": [\n'
        "    {\n"
        '      \"sentence_text\": \"la oración original\",\n'
        f'      \"detected_tenses\": [valores de: {", ".join(ALLOWED_TENSES)}],\n'
        f'      \"error_types\": [valores de: {", ".join(ALLOWED_ERROR_TYPES)}],\n'
        f'      \"topics\": [valores de: {", ".join(ALLOWED_TOPICS)}],\n'
        '      \"notable_vocabulary\": [\n'
        "        {\n"
        '          \"word\": \"palabra\",\n'
        f'          \"category\": \"uno de: {", ".join(ALLOWED_VOCAB_CATEGORIES)}\",\n'
        '          \"english_gloss\": \"significado en inglés (opcional)\"\n'
        "        }\n"
        "      ],\n"
        '      \"feedback\": \"comentario corto en español sobre esa oración\"\n'
        "    }\n"
        "  ],\n"
        f'  \"overall_error_types\": [lista de {", ".join(ALLOWED_ERROR_TYPES)} que más aparecen],\n'
        f'  \"overall_topics\": [lista de {", ".join(ALLOWED_TOPICS)} que mejor describen el mensaje],\n'
        '  \"key_takeaways\": \"consejo general en español\"\n'
        "}\n\n"
        "Reglas IMPORTANTES:\n"
        "- Usa SOLO las cadenas permitidas exactamente (mismas minúsculas) para error_types, detected_tenses, topics y category.\n"
        "- Si no hay errores o temas claros, usa [] (array vacío).\n"
        "- Responde ÚNICAMENTE con el JSON. Ningún texto antes o después."
    )

    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=800,
        )
    except Exception as err:  # noqa: BLE001
        st.warning(f"No se pudo analizar el mensaje con gpt-5-mini: {err}")
        return build_fallback_analysis(user_text, reason=str(err))

    analysis_text = response.choices[0].message.content if response.choices else None

    if not analysis_text:
        st.info("El modelo no devolvió contenido. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="empty_response")

    try:
        parsed = json.loads(analysis_text)
        if isinstance(parsed, dict):
            parsed.setdefault("source", "gpt")
        return parsed
    except json.JSONDecodeError as err:
        st.warning(
            f"El análisis devuelto no estaba en JSON válido ({err}). "
            "Usando análisis heurístico local."
        )
        return build_fallback_analysis(user_text, reason="json_decode_error")


def record_message_analysis(user_text: str, analysis_result):
    """Store per-message analytics in session state and update vocab cache."""
    entry = {
        "user_text": user_text,
        "analysis": analysis_result,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    st.session_state["message_analytics"].append(entry)

    if not analysis_result:
        return

    vocab_terms = set()
    for sentence in analysis_result.get("sentences", []):
        for vocab_item in sentence.get("notable_vocabulary", []):
            word = vocab_item.get("word")
            if word:
                vocab_terms.add(word.lower())

    if vocab_terms:
        st.session_state["user_vocabulary"].update(vocab_terms)


def build_fallback_analysis(user_text: str, reason: str | None = None):
    """Provide a heuristic analysis when GPT evaluation is unavailable."""
    cleaned = user_text.strip()
    if not cleaned:
        cleaned = "Mensaje vacío"
    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
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
        sentence_entries.append(
            {
                "sentence_text": sentence,
                "detected_tenses": [],
                "error_types": ["none"],
                "topics": ["other"],
                "notable_vocabulary": vocab_items,
                "feedback": "Buen trabajo. Esta retroalimentación es generada automáticamente.",
            }
        )

    analysis = {
        "message_summary": cleaned[:120],
        "sentences": sentence_entries,
        "overall_error_types": ["none"],
        "overall_topics": ["other"],
        "key_takeaways": (
            "Seguimos recopilando datos; esta evaluación se generó con heurísticas."
        ),
        "source": "fallback",
    }
    if reason:
        analysis["fallback_reason"] = reason
    return analysis


