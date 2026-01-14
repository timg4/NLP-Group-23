#!/usr/bin/env python3
"""Main entry point for Direct OpenAI API NER."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths for imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if os.path.basename(PROJECT_ROOT) == "models":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
# Add rule-chef-v2 first (for data_loader), then THIS_DIR (takes priority for evaluate)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "rule-chef-v2"))
sys.path.insert(0, THIS_DIR)

from openai import OpenAI
from data_loader import load_data
from extractor import extract, LABELS
from evaluate import evaluate, print_results

# Number of parallel API calls
MAX_WORKERS = 10

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "manual_annotation", "hand_labelled.conllu")


def main():
    print("=" * 60)
    print("Direct OpenAI API NER")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    data = load_data(DATA_PATH)
    print(f"  Loaded {len(data)} sentences")

    # Count entities per label
    label_counts = {label: 0 for label in LABELS}
    for _, spans in data:
        for sp in spans:
            if sp["label"] in label_counts:
                label_counts[sp["label"]] += 1
    for label, count in label_counts.items():
        print(f"  {label}: {count} gold entities")

    # 2. Extract with OpenAI (parallel)
    print(f"\n[2] Extracting entities with OpenAI API ({MAX_WORKERS} parallel workers)...")
    client = OpenAI()

    # Prepare indexed tasks
    texts = [text for text, _ in data]
    predictions: list = [None] * len(texts)  # type: ignore

    def process(idx_text):
        idx, text = idx_text
        return idx, extract(text, client)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process, (i, t)) for i, t in enumerate(texts)]
        for future in as_completed(futures):
            idx, spans = future.result()
            predictions[idx] = spans
            completed += 1
            if completed % 25 == 0 or completed == len(texts):
                print(f"  Processed {completed}/{len(texts)}...")

    print(f"  Done!")

    # 3. Sample predictions
    print("\n[3] Sample predictions:")
    for i in range(min(5, len(data))):
        text, gold = data[i]
        pred = predictions[i]
        print(f"\n  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        if gold:
            print(f"  Gold: {[(s['label'], s['text']) for s in gold]}")
        if pred:
            print(f"  Pred: {[(s['label'], s['text']) for s in pred]}")

    # 4. Evaluate
    print("\n[4] Evaluation...")

    # Exact match
    results_exact = evaluate(data, predictions, match_mode="exact")
    print_results(results_exact, "F1 Scores (Exact Match)")

    # Overlap match (more lenient)
    results_overlap = evaluate(data, predictions, match_mode="overlap")
    print_results(results_overlap, "F1 Scores (Overlap Match)")

    # 5. Show all errors
    print("\n[5] Error Analysis (Exact Match)...")
    show_errors(data, predictions)

    print("=" * 60)
    print("Done!")
    print("=" * 60)


def show_errors(data, predictions, match_mode="exact"):
    """Show all prediction errors: false positives and false negatives."""

    def spans_match(pred, gold, mode):
        if mode == "exact":
            return pred["start"] == gold["start"] and pred["end"] == gold["end"]
        else:  # overlap
            return not (pred["end"] <= gold["start"] or pred["start"] >= gold["end"])

    fp_count = 0  # false positives
    fn_count = 0  # false negatives

    print("\n" + "=" * 60)
    print("FALSE NEGATIVES (missed gold entities)")
    print("=" * 60)

    for i, ((text, gold_spans), pred_spans) in enumerate(zip(data, predictions)):
        # Find false negatives (gold entities not matched by any prediction)
        for gold in gold_spans:
            matched = False
            for pred in pred_spans:
                if pred["label"] == gold["label"] and spans_match(pred, gold, match_mode):
                    matched = True
                    break
            if not matched:
                fn_count += 1
                # Check if there's an overlapping prediction
                overlapping = [p for p in pred_spans if p["label"] == gold["label"]
                              and not (p["end"] <= gold["start"] or p["start"] >= gold["end"])]
                print(f"\n[{gold['label']}] Missed: '{gold['text']}' [{gold['start']}:{gold['end']}]")
                print(f"  Text: ...{text[max(0,gold['start']-20):gold['end']+20]}...")
                if overlapping:
                    print(f"  Overlapping pred: {[(p['text'], p['start'], p['end']) for p in overlapping]}")

    print(f"\nTotal false negatives: {fn_count}")

    print("\n" + "=" * 60)
    print("FALSE POSITIVES (wrong predictions)")
    print("=" * 60)

    for i, ((text, gold_spans), pred_spans) in enumerate(zip(data, predictions)):
        # Find false positives (predictions not matched by any gold)
        for pred in pred_spans:
            matched = False
            for gold in gold_spans:
                if pred["label"] == gold["label"] and spans_match(pred, gold, match_mode):
                    matched = True
                    break
            if not matched:
                fp_count += 1
                # Check if there's an overlapping gold
                overlapping = [g for g in gold_spans if g["label"] == pred["label"]
                              and not (g["end"] <= pred["start"] or g["start"] >= pred["end"])]
                print(f"\n[{pred['label']}] Wrong: '{pred['text']}' [{pred['start']}:{pred['end']}]")
                print(f"  Text: ...{text[max(0,pred['start']-20):pred['end']+20]}...")
                if overlapping:
                    print(f"  Overlapping gold: {[(g['text'], g['start'], g['end']) for g in overlapping]}")

    print(f"\nTotal false positives: {fp_count}")


if __name__ == "__main__":
    main()
