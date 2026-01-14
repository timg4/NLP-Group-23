#!/usr/bin/env python3
"""
Run milestone2 TokenNB (uniform priors) on stratified 80/20 split.
"""
import argparse
import os
import sys
import time
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "milestone2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef-v2"))

from milestone2 import TokenNB, build_token_label_pairs
from common import (
    collapse_bio,
    prepare_split,
    print_split_counts,
    save_predictions_conllu,
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
        default=os.path.join(THIS_DIR, "results", "TokenNB_uniform_priors"),
        help="Directory for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_set, dev_set, split_counts = prepare_split(args.data)
    print_split_counts(split_counts)

    train_pairs = build_token_label_pairs(
        [{"tokens": s["token_forms"], "labels": s["labels"]} for s in train_set]
    )

    model = TokenNB(use_uniform_priors=True)
    start = time.perf_counter()
    model.count_tokens(train_pairs)
    model.calculate_weights()

    gold_labels = []
    pred_labels = []
    sentence_predictions = []

    for sent in dev_set:
        pred = model.predict_sentence(sent["token_forms"])
        collapsed_gold = collapse_bio(sent["labels"])
        collapsed_pred = collapse_bio(pred)
        gold_labels.append(collapsed_gold)
        pred_labels.append(collapsed_pred)
        sentence_predictions.append(
            {"tokens": sent["token_forms"], "gold": collapsed_gold, "pred": collapsed_pred}
        )

    elapsed = time.perf_counter() - start
    report_text, report_dict = evaluate_predictions(gold_labels, pred_labels)

    out_dir = Path(args.results_dir)
    write_metrics(out_dir, "TokenNB_uniform_priors", report_text, report_dict, elapsed)
    save_predictions_conllu(sentence_predictions, out_dir / "predictions.conllu")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
