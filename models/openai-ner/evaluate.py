"""Evaluation with precision, recall, F1 for each entity type."""
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directory to path for imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if os.path.basename(PROJECT_ROOT) == "models":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "rule-chef-v2"))

from extractor import LABELS


def span_match(pred: Dict, gold: Dict, mode: str = "exact") -> bool:
    """Check if predicted span matches gold span.

    Args:
        pred: Predicted span {"start": X, "end": Y, "text": "..."}
        gold: Gold span {"start": X, "end": Y, "text": "..."}
        mode: "exact" for exact match, "overlap" for any overlap
    """
    if mode == "exact":
        return pred["start"] == gold["start"] and pred["end"] == gold["end"]
    elif mode == "overlap":
        return not (pred["end"] <= gold["start"] or pred["start"] >= gold["end"])
    return False


def evaluate(
    data: List[Tuple[str, List[Dict]]],
    predictions: List[List[Dict]],
    match_mode: str = "exact"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions against gold standard.

    Args:
        data: List of (text, gold_spans) tuples
        predictions: List of predicted span lists (parallel to data)
        match_mode: "exact" or "overlap"

    Returns: {
        "ORG": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 35},
        "MON": {...},
        "LEG": {...},
        "micro": {...},
        "macro": {...}
    }
    """
    # Counters per label
    tp = defaultdict(int)  # true positives
    fp = defaultdict(int)  # false positives
    fn = defaultdict(int)  # false negatives

    for (text, gold_spans), pred_spans in zip(data, predictions):
        # Group by label
        gold_by_label = defaultdict(list)
        pred_by_label = defaultdict(list)

        for sp in gold_spans:
            gold_by_label[sp["label"]].append(sp)
        for sp in pred_spans:
            pred_by_label[sp["label"]].append(sp)

        # Evaluate each label
        for label in LABELS:
            golds = gold_by_label[label]
            preds = pred_by_label[label]

            # Track which golds have been matched
            matched_golds = set()

            for pred in preds:
                matched = False
                for i, gold in enumerate(golds):
                    if i not in matched_golds and span_match(pred, gold, match_mode):
                        tp[label] += 1
                        matched_golds.add(i)
                        matched = True
                        break
                if not matched:
                    fp[label] += 1

            # Unmatched golds are false negatives
            fn[label] += len(golds) - len(matched_golds)

    # Calculate metrics
    results = {}

    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[label] + fn[label],  # total gold spans
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }

    # Micro average (aggregate all spans)
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    results["micro"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "support": total_tp + total_fn,
    }

    # Macro average (average of per-label metrics)
    macro_precision = sum(results[l]["precision"] for l in LABELS) / len(LABELS)
    macro_recall = sum(results[l]["recall"] for l in LABELS) / len(LABELS)
    macro_f1 = sum(results[l]["f1"] for l in LABELS) / len(LABELS)

    results["macro"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }

    return results


def print_results(results: Dict[str, Dict[str, float]], title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    print(f"\n{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 52)

    for label in LABELS:
        r = results[label]
        print(f"{label:<10} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['support']:>10}")

    print("-" * 52)

    r = results["micro"]
    print(f"{'micro avg':<10} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['support']:>10}")

    r = results["macro"]
    print(f"{'macro avg':<10} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f}")

    print()
