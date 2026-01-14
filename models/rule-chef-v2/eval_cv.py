#!/usr/bin/env python3
"""5-fold cross validation for RuleChef NER v2."""
import os
import sys
import random
from statistics import mean
from typing import Dict, List, Tuple

# Add this directory to path for local imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if os.path.basename(PROJECT_ROOT) == "models":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef", "rulechef"))

from data_loader import load_data
from trainer import train_all, LABELS
from inference import predict

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "manual_annotation", "hand_labelled.conllu")
BASE_STORAGE = os.path.join(THIS_DIR, "rulechef_v2_data")


def score_spans(gold: List[Dict], pred: List[Dict], labels=("ORG", "MON", "LEG")) -> Dict:
    def _to_keys(spans: List[Dict]):
        return {(sp["label"], sp["start"], sp["end"]) for sp in spans}

    def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    out = {"by_label": {}, "micro": {}}
    g_all = _to_keys(gold)
    p_all = _to_keys(pred)

    tp = len(g_all & p_all)
    fp = len(p_all - g_all)
    fn = len(g_all - p_all)
    out["micro"] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    for lab in labels:
        g = {(l, s, e) for (l, s, e) in g_all if l == lab}
        p = {(l, s, e) for (l, s, e) in p_all if l == lab}
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        out["by_label"][lab] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return out


def dump_mismatches(
    test_items: List[Tuple[str, List[Dict]]],
    preds_by_item: List[List[Dict]],
    max_items: int = 5,
):
    """
    Print a small sample of false negatives/positives per label.
    """
    labels = ["ORG", "MON", "LEG"]
    fn = {lab: [] for lab in labels}
    fp = {lab: [] for lab in labels}

    def _key(sp: Dict):
        return (sp["label"], sp["start"], sp["end"])

    for (text, gold_spans), pred_spans in zip(test_items, preds_by_item):
        gset = {_key(sp) for sp in gold_spans}
        pset = {_key(sp) for sp in pred_spans}

        for sp in gold_spans:
            if _key(sp) not in pset:
                fn[sp["label"]].append((text, sp))

        for sp in pred_spans:
            if _key(sp) not in gset:
                fp[sp["label"]].append((text, sp))

    def _print_block(title: str, items: List[Tuple[str, Dict]]):
        print("\n" + title)
        print("-" * len(title))
        for text, sp in items[:max_items]:
            print(f"span='{sp.get('text', '')}' [{sp['start']}:{sp['end']}]")
            print(f"sent: {text}")

    for lab in labels:
        _print_block(f"FALSE NEGATIVES ({lab})", fn[lab])
        _print_block(f"FALSE POSITIVES ({lab})", fp[lab])


def make_folds(n_items: int, k: int = 5, seed: int = 1337) -> List[List[int]]:
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return [indices[i::k] for i in range(k)]


def build_by_label(items: List[Tuple[str, List[Dict]]]) -> Dict[str, List[Tuple[str, List[Dict]]]]:
    data = {lab: [] for lab in LABELS}
    for text, spans in items:
        for label in LABELS:
            label_spans = [
                {"start": sp["start"], "end": sp["end"], "text": sp["text"]}
                for sp in spans
                if sp["label"] == label
            ]
            data[label].append((text, label_spans))
    return data


def main():
    data = load_data(DATA_PATH)
    folds = make_folds(len(data), k=5, seed=1337)

    fold_micro_f1 = []
    fold_label_f1 = {lab: [] for lab in LABELS}

    for i, test_idx in enumerate(folds[:1]):
        fold_name = f"fold_{i}"
        test_idx = set(test_idx)
        train_items = [item for j, item in enumerate(data) if j not in test_idx]
        test_items = [item for j, item in enumerate(data) if j in test_idx]

        print("\n" + "=" * 60)
        print(f"CV {fold_name}: train={len(train_items)} test={len(test_items)}")
        print("=" * 60)

        train_by_label = build_by_label(train_items)
        storage_path = os.path.join(BASE_STORAGE, fold_name)

        chefs = train_all(
            train_by_label,
            storage_path=storage_path,
            dataset_prefix=fold_name,
            skip_if_exists=True,
        )

        gold_all = []
        pred_all = []
        preds_by_item = []
        for text, gold_spans in test_items:
            preds = predict(text, chefs)
            gold_all.extend(gold_spans)
            pred_all.extend(preds)
            preds_by_item.append(preds)

        scores = score_spans(gold_all, pred_all)
        micro_f1 = scores["micro"]["f1"]
        fold_micro_f1.append(micro_f1)
        for lab in LABELS:
            fold_label_f1[lab].append(scores["by_label"][lab]["f1"])

        print(
            f"{fold_name}: micro_f1={micro_f1:.4f} "
            f"ORG={scores['by_label']['ORG']['f1']:.4f} "
            f"MON={scores['by_label']['MON']['f1']:.4f} "
            f"LEG={scores['by_label']['LEG']['f1']:.4f}"
        )
        dump_mismatches(test_items, preds_by_item, max_items=5)

    print("\nMEAN over folds:")
    print(f"micro_f1={mean(fold_micro_f1):.4f}")
    for lab in LABELS:
        print(f"{lab}_f1={mean(fold_label_f1[lab]):.4f}")


if __name__ == "__main__":
    main()
