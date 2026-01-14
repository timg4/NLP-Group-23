"""Direct OpenAI API extraction for NER."""
import json
from typing import Dict, List
from openai import OpenAI

LABELS = ["ORG", "MON", "LEG"]

SYSTEM_PROMPT = """You are a Named Entity Recognition system for German financial documents.
Extract entities of three types:
- ORG: Organization/Company names (e.g., DZ BANK, Berlin Hyp, NORD/LB)
- MON: Monetary amounts with currency (e.g., EUR 4.899.938, 2,4 Mrd., USD 500 Mio.)
- LEG: Legal references (e.g., Abs. 271, Nr. 5, HGB, BGB, AktG)

Return JSON with "entities" array. Each entity has:
- "label": ORG, MON, or LEG
- "text": the EXACT text as it appears in the input (copy-paste exactly)

Rules:
1. ORG: Look for legal forms (AG, GmbH, Bank) or known financial institutions
2. MON: Must have BOTH a number AND currency marker (EUR, USD, €, Mio., Mrd.)
3. LEG: Must have legal markers (§, Abs., Nr., Art.) or law abbreviations
4. German capitalizes all nouns - don't extract generic nouns like "Kapital"
5. The "text" field must be an EXACT substring of the input - copy it precisely
6. If no entities found, return {"entities": []}
"""


def extract(text: str, client: OpenAI, model: str = "gpt-5-mini") -> List[Dict]:
    """
    Extract NER spans from text using OpenAI API.

    Args:
        text: Input text to analyze
        client: OpenAI client instance
        model: Model to use (default: gpt-5-mini)

    Returns:
        List of span dicts: [{"label": "ORG", "start": X, "end": Y, "text": "..."}, ...]
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities from this text:\n\n{text}"}
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    if not content:
        return []

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return []

    entities = result.get("entities", [])
    if not isinstance(entities, list):
        return []

    # Find each entity in the text and compute offsets ourselves
    validated = []
    used_positions = set()  # Track used positions to avoid duplicates

    for entity in entities:
        try:
            label = entity.get("label", "")
            entity_text = entity.get("text", "")
        except (KeyError, TypeError):
            continue

        if label not in LABELS:
            continue
        if not entity_text or len(entity_text) < 2:
            continue

        # Find all occurrences of this text in the input
        start = 0
        while True:
            idx = text.find(entity_text, start)
            if idx == -1:
                break

            end = idx + len(entity_text)

            # Check if this position is already used
            if idx not in used_positions:
                validated.append({
                    "label": label,
                    "start": idx,
                    "end": end,
                    "text": entity_text
                })
                used_positions.add(idx)
                break  # Use first occurrence

            start = idx + 1  # Try next occurrence

    return validated


def extract_batch(texts: List[str], client: OpenAI, model: str = "gpt-5-mini") -> List[List[Dict]]:
    """Extract NER spans for multiple texts."""
    return [extract(text, client, model) for text in texts]


if __name__ == "__main__":
    # Quick test
    client = OpenAI()
    test_text = "Die DZ BANK erzielte EUR 2,4 Mrd. gemäß § 271 HGB."
    spans = extract(test_text, client)

    print(f"Text: {test_text}")
    print(f"Spans:")
    for sp in spans:
        print(f"  {sp['label']}: '{sp['text']}' [{sp['start']}:{sp['end']}]")
