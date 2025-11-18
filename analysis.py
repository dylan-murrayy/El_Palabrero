import datetime
import json
import re
from enum import Enum
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel

from vocabulary import extract_words


# Use a stable chat model that works with the chat.completions API.
ANALYSIS_MODEL = "gpt-4.1-mini"


# Enums for Structured Outputs
class ErrorType(str, Enum):
    NONE = "none"
    GRAMMAR = "grammar"
    CONJUGATION = "conjugation"
    AGREEMENT = "agreement"
    VOCABULARY = "vocabulary"
    ORTHOGRAPHY = "orthography"
    PUNCTUATION = "punctuation"
    REGISTER = "register"
    PRONUNCIATION = "pronunciation"
    OTHER = "other"

class Tense(str, Enum):
    PRESENT = "present"
    PRETERITE = "preterite"
    IMPERFECT = "imperfect"
    FUTURE = "future"
    CONDITIONAL = "conditional"
    PRESENT_PERFECT = "present-perfect"
    PAST_PERFECT = "past-perfect"
    FUTURE_PERFECT = "future-perfect"
    CONDITIONAL_PERFECT = "conditional-perfect"
    IMPERATIVE = "imperative"
    PRESENT_SUBJUNCTIVE = "present-subjunctive"
    IMPERFECT_SUBJUNCTIVE = "imperfect-subjunctive"
    OTHER = "other"

class Topic(str, Enum):
    EVERYDAY_LIFE = "everyday-life"
    TRAVEL = "travel"
    WORK_AND_STUDIES = "work-and-studies"
    FOOD_AND_COOKING = "food-and-cooking"
    EMOTIONS = "emotions"
    CULTURE_AND_ARTS = "culture-and-arts"
    TECHNOLOGY = "technology"
    HEALTH_AND_WELLNESS = "health-and-wellness"
    RELATIONSHIPS = "relationships"
    CURRENT_EVENTS = "current-events"
    HOBBIES = "hobbies"
    OTHER = "other"

class VocabCategory(str, Enum):
    NEW_WORD = "new-word"
    ADVANCED_WORD = "advanced-word"
    REVIEW_WORD = "review-word"
    IDIOM = "idiom"
    COLLOCATION = "collocation"
    OTHER = "other"

# Pydantic Models for Structured Outputs
class VocabularyItem(BaseModel):
    word: str
    category: VocabCategory
    english_gloss: Optional[str] = None

class SentenceAnalysis(BaseModel):
    sentence_text: str
    detected_tenses: List[Tense]
    error_types: List[ErrorType]
    topics: List[Topic]
    notable_vocabulary: List[VocabularyItem]
    feedback: str

class MessageAnalysis(BaseModel):
    message_summary: str
    sentences: List[SentenceAnalysis]
    overall_error_types: List[ErrorType]
    overall_topics: List[Topic]
    key_takeaways: str


ALLOWED_ERROR_TYPES = [e.value for e in ErrorType]
ALLOWED_TENSES = [t.value for t in Tense]
ALLOWED_TOPICS = [t.value for t in Topic]
ALLOWED_VOCAB_CATEGORIES = [c.value for c in VocabCategory]


def analyze_user_message(user_text: str):
    """Call OpenAI chat completions and parse JSON analysis using Structured Outputs."""
    client = st.session_state.get("openai_client")
    if not user_text.strip():
        return None
    if client is None:
        st.info("Analítica GPT no disponible. Usando análisis heurístico local.")
        return build_fallback_analysis(user_text, reason="no_client")

    system_message = (
        "Eres Palabrero, un tutor de español.\n"
        "TAREA: Analiza el mensaje del estudiante."
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_text},
            ],
            response_format=MessageAnalysis,
        )
        
        analysis_result = completion.choices[0].message.parsed
        
        # Convert Pydantic model to dict for compatibility with existing code
        if analysis_result:
            parsed = analysis_result.model_dump(mode='json')
            parsed["source"] = "gpt"
            return parsed
        else:
             return build_fallback_analysis(user_text, reason="empty_parsed_response")

    except Exception as err:  # noqa: BLE001
        st.warning(f"No se pudo analizar el mensaje con {ANALYSIS_MODEL}: {err}")
        return build_fallback_analysis(user_text, reason=str(err))



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


