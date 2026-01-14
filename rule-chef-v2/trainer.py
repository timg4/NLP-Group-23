"""Train RuleChef models for NER - minimal configuration."""
import os
import random
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

TASK_DESCRIPTIONS = {
    "ORG": "Extract organization names from German financial text.",
    "MON": "Extract monetary amounts (with currency markers) from German financial text.",
    "LEG": "Extract legal references from German financial text.",
}

# ONE critical feedback per label - only the most important constraint

# Label-specific training configuration
LABEL_CONFIG = {
    "ORG": {
        "allowed_formats": [RuleFormat.CODE, RuleFormat.REGEX],
        "sampling_strategy": "varied",
        "max_refinement_iterations": 3,
        "feedbacks": [
            "Don't match generic capitalized nouns - German capitalizes all nouns.",
            "Organization names often include legal forms or suffixes (AG, GmbH, KG, SE, Bank, Sparkasse, e.V.).",
        ],
    },
    "MON": {
        "allowed_formats": [RuleFormat.REGEX, RuleFormat.CODE],
        "sampling_strategy": "varied",
        "max_refinement_iterations": 2,
        "feedbacks": [
            "Should contain a digit AND a currency marker (EUR, USD, Mio., Mrd., $, GBP).",
            "Avoid law/section references without currency markers.",
        ],
    },
    "LEG": {
        "allowed_formats": [RuleFormat.REGEX, RuleFormat.CODE],
        "sampling_strategy": "varied",
        "max_refinement_iterations": 3,
        "feedbacks": [
            "Should include legal markers (e.g. Abs., Nr., Art., Section) or law abbreviations (BGB, HGB, AktG).",
            "Capture full legal references like Section 433 BGB or Art. 15 DSGVO.",
        ],
    },
}


def create_task(label: str) -> Task:
    """Create a minimal Task for a label."""
    return Task(
        name=f"extract_{label}",
        description=TASK_DESCRIPTIONS[label],
        input_schema={"text": "string", "context": "string"},
        output_schema={"spans": "List[Span]"},
        type=TaskType.EXTRACTION,
    )


def train_model(
    data: List[Tuple[str, List[Dict]]],
    label: str,
    model_name: str = "gpt-4o-mini",
    storage_path: str = "./rulechef_v2_data",
    dataset_name: Optional[str] = None,
    skip_if_exists: bool = False,
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
    config = LABEL_CONFIG[label]
    client = OpenAI()

    chef = RuleChef(
        task=create_task(label),
        client=client,
        dataset_name=dataset_name or f"ner_{label}",
        storage_path=storage_path,
        allowed_formats=config["allowed_formats"],
        sampling_strategy=config["sampling_strategy"],
        model=model_name,
        auto_trigger=False,
    )

    if skip_if_exists and chef.dataset.rules:
        print(
            f"Skipping training for {label}; loaded {len(chef.dataset.rules)} rules."
        )
        return chef

    # Add all training examples
    rng = random.Random(13)
    shuffled_data = list(data)
    rng.shuffle(shuffled_data)
    for text, spans in shuffled_data:
        chef.add_example(
            input_data={"text": text, "context": text},
            output_data={"spans": spans},
        )

    # Add label-specific feedback
    for feedback in config["feedbacks"]:
        chef.add_feedback(feedback)

    # Learn rules
    print(f"Learning rules for {label}...")
    chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=config["max_refinement_iterations"],
        sampling_strategy=config["sampling_strategy"],
    )

    # Print learned rules
    print(f"Learned {len(chef.dataset.rules)} rules for {label}:")
    for r in chef.dataset.rules:
        print(f"  - {r.name} (priority: {r.priority}, confidence: {r.confidence:.2f})")

    return chef


def train_all(
    data_by_label: Dict[str, List[Tuple[str, List[Dict]]]],
    model_name: str = "gpt-4o-mini",
    storage_path: str = "./rulechef_v2_data",
    dataset_prefix: Optional[str] = None,
    skip_if_exists: bool = False,
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
            dataset_name=f"{dataset_prefix}_{label}" if dataset_prefix else None,
            skip_if_exists=skip_if_exists,
        )

    return chefs


if __name__ == "__main__":
    from .data_loader import load_data_by_label

    data = load_data_by_label("../data/manual_annotation2/my_labels.conllu")
    chefs = train_all(data)
