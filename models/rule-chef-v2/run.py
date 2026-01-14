#!/usr/bin/env python3
"""Main entry point for RuleChef NER v2."""
import os
import sys

# Add this directory to path for local imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if os.path.basename(PROJECT_ROOT) == "models":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef", "rulechef"))

from data_loader import load_data_by_label, load_data
from trainer import train_all, LABELS
from inference import predict
from evaluate import evaluate, print_results

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "manual_annotation", "hand_labelled.conllu")
STORAGE_PATH = os.path.join(THIS_DIR, "rulechef_v2_data")


def main():
    print("=" * 60)
    print("RuleChef NER v2 - Minimal Implementation")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    data = load_data_by_label(DATA_PATH)

    for label in LABELS:
        n_pos = sum(1 for _, spans in data[label] if spans)
        n_neg = sum(1 for _, spans in data[label] if not spans)
        print(f"  {label}: {n_pos} positive, {n_neg} negative sentences")

    # 2. Train models
    print("\n[2] Training models...")
    chefs = train_all(data, storage_path=STORAGE_PATH)

    # 3. Test inference
    print("\n[3] Testing inference...")
    test_sentences = [
        "Die DZ BANK erzielte EUR 2,4 Mrd.",
        "Der Vorstand der NORD/LB hat sich an dem Treffen beteiligt.",
        "Dies gilt nach Abs. 271 HGB und Art. 15 der Verordnung.",
        "Die Berlin Hyp verzeichnete Einnahmen von USD 500 Mio.",
    ]

    for text in test_sentences:
        spans = predict(text, chefs)
        print(f"\nText: {text}")
        if spans:
            for sp in spans:
                print(f"  {sp['label']}: '{sp['text']}' [{sp['start']}:{sp['end']}]")
        else:
            print("  (no entities found)")

    # 4. Evaluate with F1 scores
    print("\n[4] Evaluating on full dataset...")
    full_data = load_data(DATA_PATH)
    results = evaluate(full_data, chefs, match_mode="exact")
    print_results(results, "F1 Scores (Exact Match)")

    print("\n" + "=" * 60)
    print("Done! Models saved to:", STORAGE_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
