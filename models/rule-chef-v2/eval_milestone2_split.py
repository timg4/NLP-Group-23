#!/usr/bin/env python3
"""
Train RuleChef on an 80/20 split and evaluate with milestone2-style metrics.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import classification_report

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if os.path.basename(PROJECT_ROOT) == "models":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef", "rulechef"))

from data_loader import _align_tokens, _tokens_to_spans
from inference import predict
from trainer import LABELS, train_all
from stratified_split import SPLIT_SEED, stratified_split


LABEL_ORDER = ["B-LEG", "I-LEG", "B-MON", "I-MON", "B-ORG", "I-ORG", "O"]


def load_conllu_with_text(path: str) -> List[Dict]:
    """
    Load a CoNLL-U file with BIO labels and keep # text if present.
    """
    sentences = []
    tokens, labels = [], []
    sent_text = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    text = sent_text or " ".join(tokens)
                    spans = _tokens_to_spans(list(zip(tokens, labels)), text)
                    sentences.append(
                        {
                            "text": text,
                            "tokens": tokens,
                            "labels": labels,
                            "spans": spans,
                        }
                    )
                    tokens, labels, sent_text = [], [], None
                continue

            if line.startswith("# text ="):
                sent_text = line.split("=", 1)[1].strip()
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            tokens.append(parts[1])
            labels.append(parts[-1])

    if tokens:
        text = sent_text or " ".join(tokens)
        spans = _tokens_to_spans(list(zip(tokens, labels)), text)
        sentences.append(
            {
                "text": text,
                "tokens": tokens,
                "labels": labels,
                "spans": spans,
            }
        )

    return sentences




def build_data_by_label(items: List[Dict]) -> Dict[str, List]:
    data_by_label = {lab: [] for lab in LABELS}
    for item in items:
        text = item["text"]
        spans = item["spans"]
        for label in LABELS:
            label_spans = [
                {"start": sp["start"], "end": sp["end"], "text": sp["text"]}
                for sp in spans
                if sp["label"] == label
            ]
            data_by_label[label].append((text, label_spans))
    return data_by_label


def spans_to_bio(spans: List[Dict], tokens: List[str], text: str) -> List[str]:
    aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        text = " ".join(tokens)
        aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        return ["O"] * len(tokens)

    offsets = [(start, end) for _, start, end, _ in aligned]
    labels = ["O"] * len(tokens)

    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        label = span["label"]
        s_start = span["start"]
        s_end = span["end"]
        in_span = False
        for i, (t_start, t_end) in enumerate(offsets):
            if t_end <= s_start or t_start >= s_end:
                if in_span:
                    in_span = False
                continue
            if labels[i] != "O":
                continue
            labels[i] = f"B-{label}" if not in_span else f"I-{label}"
            in_span = True

    return labels


def save_predictions_conllu(sentence_predictions: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sent_pred in enumerate(sentence_predictions):
            f.write(f"# sent_id = {i + 1}\n")
            for j, (token, label) in enumerate(zip(sent_pred["tokens"], sent_pred["pred"])):
                f.write(f"{j+1}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\t{label}\n")
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "manual_annotation", "hand_labelled.conllu"),
        help="Single CoNLL-U file, will be split 80/20 into train/dev.",
    )
    parser.add_argument(
        "--storage",
        default=os.path.join(THIS_DIR, "rulechef_v2_data"),
        help="Storage path for RuleChef datasets.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(THIS_DIR, "results_milestone2"),
        help="Directory for metrics and predictions output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_sents = load_conllu_with_text(args.data)
    train_set, dev_set, split_counts = stratified_split(
        all_sents, train_ratio=0.8, seed=SPLIT_SEED
    )

    print(f"Total sentences: {len(all_sents)}")
    print(f"Train sentences: {len(train_set)}")
    print(f"Dev sentences:   {len(dev_set)}")
    print("Stratified split by sentence label:")
    for key in sorted(split_counts.keys()):
        c = split_counts[key]
        print(f"  {key}: total={c['total']}, train={c['train']}, dev={c['dev']}")

    data_by_label = build_data_by_label(train_set)
    chefs = train_all(data_by_label, storage_path=args.storage)

    y_true, y_pred = [], []
    sentence_predictions = []

    for sent in dev_set:
        gold = sent["labels"]
        pred_spans = predict(sent["text"], chefs)
        pred = spans_to_bio(pred_spans, sent["tokens"], sent["text"])
        if len(pred) != len(gold):
            raise ValueError("Length mismatch between gold and prediction")
        y_true.extend(gold)
        y_pred.extend(pred)
        sentence_predictions.append(
            {"tokens": sent["tokens"], "gold": gold, "pred": pred}
        )

    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=3)
    print("\n=== RuleChef ===")
    print(report)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics_summary.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RULECHEF MILESTONE 2 METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("=== RuleChef ===\n")
        f.write(report + "\n")

    preds_path = results_dir / "rulechef_predictions.conllu"
    save_predictions_conllu(sentence_predictions, preds_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()

