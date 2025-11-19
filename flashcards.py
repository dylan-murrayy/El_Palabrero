import json
import re
from typing import List, Dict, Any

import io
import zipfile
import random

def generate_cloze_cards(conversations: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, str]]:
    """
    Analyze conversations to find mistakes and generate cloze cards.
    
    Args:
        conversations: List of conversation dictionaries containing 'analytics'.
        limit: Maximum number of cards to generate.
        
    Returns:
        List of dicts with 'front' (cloze) and 'back' (explanation).
    """
    all_cards = []
    
    for chat in conversations:
        analytics = chat.get("analytics", [])
        for entry in analytics:
            analysis = entry.get("analysis", {})
            sentences = analysis.get("sentences", [])
            
            for sentence in sentences:
                # We only want sentences with errors
                error_types = sentence.get("error_types", [])
                if not error_types or "none" in error_types:
                    continue
                
                # Skip if only punctuation error (user preference usually)
                if len(error_types) == 1 and "punctuation" in error_types:
                    continue
                
                original_text = sentence.get("sentence_text", "")
                feedback = sentence.get("feedback", "")
                
                card = {
                    "name": f"Correction: {original_text[:20]}...",
                    "content": f"""
## Sentence
{original_text}

## Feedback
{feedback}

## Errors
{', '.join(error_types)}
"""
                }
                all_cards.append(card)
    
    # Sort by most recent (assuming conversations are ordered) or shuffle?
    # User asked for "10 flashcards based on my mistakes".
    # Let's prioritize recent mistakes (end of list usually) but maybe shuffle if many?
    # For now, let's take the LAST 'limit' cards as they are likely most recent.
    if len(all_cards) > limit:
        return all_cards[-limit:]
    
    return all_cards

def create_mochi_zip(deck_name: str, cards: List[Dict[str, str]]) -> bytes:
    """
    Convert a list of cards into a Mochi .mochi (ZIP) file.
    
    A .mochi file is a ZIP containing a 'data.edn' file.
    Structure of data.edn:
    {:decks [{:id "deck-id" :name "Deck Name" :cards [{:content "..."}]}]}
    """
    # Construct EDN string
    edn_lines = []
    edn_lines.append("{:decks [{")
    edn_lines.append(f'  :name "{deck_name}"')
    edn_lines.append('  :cards [')
    
    for card in cards:
        content = card["content"].replace('"', '\\"')  # Escape quotes
        edn_lines.append('    {:content "' + content + '"}')
        
    edn_lines.append('  ]')
    edn_lines.append('}]}')
    
    edn_content = "\n".join(edn_lines)
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.edn", edn_content)
    
    return zip_buffer.getvalue()
