"""Inference with trained RuleChef NER models."""
import os
import sys

# Ensure rulechef is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
RULECHEF_PATH = os.path.join(PROJECT_ROOT, "rule-chef", "rulechef")
if RULECHEF_PATH not in sys.path:
    sys.path.insert(0, RULECHEF_PATH)

from typing import Dict, List, Optional
from openai import OpenAI
from rulechef import RuleChef
from rulechef.core import RuleFormat

try:
    from .trainer import create_task, LABELS, LABEL_CONFIG
except Exception:
    try:
        from trainer import create_task, LABELS, LABEL_CONFIG
    except Exception:
        from trainer import create_task, LABELS
        LABEL_CONFIG = {label: {"allowed_formats": [RuleFormat.CODE]} for label in LABELS}


def load_models(storage_path: str = "./rulechef_v2_data") -> Dict[str, RuleChef]:
    """
    Load pre-trained RuleChef models.

    Args:
        storage_path: Where the models were saved during training

    Returns:
        Dict mapping label -> RuleChef instance
    """
    client = OpenAI()
    chefs = {}

    for label in LABELS:
        chef = RuleChef(
            task=create_task(label),
            client=client,
            dataset_name=f"ner_{label}",
            storage_path=storage_path,
            allowed_formats=LABEL_CONFIG[label]["allowed_formats"],
            auto_trigger=False,
        )
        chefs[label] = chef

    return chefs


def predict(text: str, chefs: Dict[str, RuleChef]) -> List[Dict]:
    """
    Predict NER spans for text.

    Args:
        text: Input text to tag
        chefs: Dict of trained RuleChef models (from load_models or train_all)

    Returns:
        List of span dicts: [{"label": "ORG", "start": X, "end": Y, "text": "..."}, ...]
    """
    all_spans = []

    for label, chef in chefs.items():
        if not chef.dataset.rules:
            continue

        # Use public extract API
        result = chef.extract({"text": text, "context": text})
        spans = result.get("spans", [])

        for span in spans:
            try:
                start = int(span["start"])
                end = int(span["end"])
            except (KeyError, ValueError, TypeError):
                continue

            # Validate span bounds
            if not (0 <= start < end <= len(text)):
                continue

            # Skip very short spans (likely noise)
            if end - start < 2:
                continue

            all_spans.append({
                "label": label,
                "start": start,
                "end": end,
                "text": text[start:end],
            })

    # Deduplicate exact matches
    seen = set()
    unique = []
    for sp in all_spans:
        key = (sp["label"], sp["start"], sp["end"])
        if key not in seen:
            unique.append(sp)
            seen.add(key)

    # Sort by position
    unique.sort(key=lambda x: (x["start"], x["end"]))

    return unique


def predict_batch(texts: List[str], chefs: Dict[str, RuleChef]) -> List[List[Dict]]:
    """Predict NER spans for multiple texts."""
    return [predict(text, chefs) for text in texts]


if __name__ == "__main__":
    # Quick test with loaded models
    chefs = load_models()

    test_text = "Die DZ BANK erzielte EUR 2,4 Mrd. gemae Abs. 271 HGB."
    spans = predict(test_text, chefs)

    print(f"Text: {test_text}")
    print(f"Spans: {spans}")
