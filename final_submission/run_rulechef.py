#!/usr/bin/env python3
"""
Run RuleChef on stratified 80/20 split and write metrics/predictions.
"""
import argparse
import os
import sys
import time
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef-v2"))

from inference import predict as rulechef_predict
from trainer import train_all
from common import (
    build_data_by_label,
    collapse_bio,
    prepare_split,
    print_split_counts,
    save_predictions_conllu,
    spans_to_bio,
    evaluate_predictions,
    write_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "manual_annotation2", "my_labels.conllu"),
        help="Single CoNLL-U file, will be split 80/20 into train/dev.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(THIS_DIR, "results", "RuleChef"),
        help="Directory for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_set, dev_set, split_counts = prepare_split(args.data)
    print_split_counts(split_counts)

    data_by_label = build_data_by_label(train_set)
    storage_path = os.path.join(PROJECT_ROOT, "rule-chef-v2", "rulechef_v2_data")

    start = time.perf_counter()
    chefs = train_all(data_by_label, storage_path=storage_path)

    gold_labels = []
    pred_labels = []
    sentence_predictions = []

    for sent in dev_set:
        spans = rulechef_predict(sent["text"], chefs)
        bio = spans_to_bio(spans, sent["token_forms"], sent["text"])
        collapsed_gold = collapse_bio(sent["labels"])
        collapsed_pred = collapse_bio(bio)
        gold_labels.append(collapsed_gold)
        pred_labels.append(collapsed_pred)
        sentence_predictions.append(
            {"tokens": sent["token_forms"], "gold": collapsed_gold, "pred": collapsed_pred}
        )

    elapsed = time.perf_counter() - start
    report_text, report_dict = evaluate_predictions(gold_labels, pred_labels)

    out_dir = Path(args.results_dir)
    write_metrics(out_dir, "RuleChef", report_text, report_dict, elapsed)
    save_predictions_conllu(sentence_predictions, out_dir / "predictions.conllu")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
