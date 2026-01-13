"""Train RuleChef models for NER - minimal configuration."""
import os
import sys

# Ensure rulechef is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
RULECHEF_PATH = os.path.join(PROJECT_ROOT, "rule-chef", "rulechef")
if RULECHEF_PATH not in sys.path:
    sys.path.insert(0, RULECHEF_PATH)

from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from rulechef import RuleChef, Task
from rulechef.core import RuleFormat, TaskType

LABELS = ["ORG", "MON", "LEG"]

# SHORT task descriptions - let examples teach
TASK_DESCRIPTIONS = {
    "ORG": "Extract organization names from German financial text.",
    "MON": "Extract monetary amounts (with currency markers) from German financial text.",
    "LEG": "Extract legal references from German financial text.",
}

# ONE critical feedback per label - only the most important constraint
FEEDBACKS = {
    "ORG": "Don't match generic capitalized nouns - German capitalizes all nouns.",
    "MON": "Must contain a digit AND a currency marker (EUR, USD, Mio., Mrd., etc.). No percentages.",
    "LEG": "Must include legal markers (e.g. Abs., Nr.) or law abbreviations (BGB, HGB, AktG).",
}


def create_task(label: str) -> Task:
    """Create a minimal Task for a label."""
    return Task(
        name=f"extract_{label}",
        description=TASK_DESCRIPTIONS[label],
        input_schema={"text": "str"},
        output_schema={"spans": "List[Span]"},
        type=TaskType.EXTRACTION,
    )


def train_model(
    data: List[Tuple[str, List[Dict]]],
    label: str,
    model_name: str = "gpt-5-mini",
    storage_path: str = "./rulechef_v2_data",
) -> RuleChef:
    """
    Train a RuleChef for one label.

    Args:
        data: List of (text, spans) tuples where spans are for this label only
        label: The label (ORG, MON, LEG)
        model_name: OpenAI model to use
        storage_path: Where to save learned rules

    Returns:
        Trained RuleChef instance
    """
    client = OpenAI()

    chef = RuleChef(
        task=create_task(label),
        client=client,
        dataset_name=f"ner_{label}",
        storage_path=storage_path,
        allowed_formats=[RuleFormat.CODE],  # CODE only - simpler
        model=model_name,
        auto_trigger=False,
    )

    # Add all training examples
    for text, spans in data:
        chef.add_example(
            input_data={"text": text},
            output_data={"spans": spans},
        )

    # Add ONE critical feedback
    chef.add_feedback(FEEDBACKS[label])

    # Learn rules
    print(f"Learning rules for {label}...")
    chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=2,
    )

    # Print learned rules
    print(f"Learned {len(chef.dataset.rules)} rules for {label}:")
    for r in chef.dataset.rules:
        print(f"  - {r.name} (priority: {r.priority}, confidence: {r.confidence:.2f})")

    return chef


def train_all(
    data_by_label: Dict[str, List[Tuple[str, List[Dict]]]],
    model_name: str = "gpt-5-mini",
    storage_path: str = "./rulechef_v2_data",
) -> Dict[str, RuleChef]:
    """Train RuleChef for all labels."""
    chefs = {}

    for label in LABELS:
        print(f"\n{'='*60}")
        print(f"TRAINING: {label}")
        print(f"{'='*60}")

        n_pos = sum(1 for _, spans in data_by_label[label] if spans)
        n_neg = sum(1 for _, spans in data_by_label[label] if not spans)
        print(f"Data: {n_pos} positive, {n_neg} negative examples")

        chefs[label] = train_model(
            data=data_by_label[label],
            label=label,
            model_name=model_name,
            storage_path=storage_path,
        )

    return chefs


if __name__ == "__main__":
    from .data_loader import load_data_by_label

    data = load_data_by_label("../data/manual_annotation2/my_labels.conllu")
    chefs = train_all(data)
