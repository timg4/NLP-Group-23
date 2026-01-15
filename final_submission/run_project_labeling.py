#!/usr/bin/env python3
"""
Run project labeling (spaCy + regex) on stratified 80/20 split.
"""
import argparse
import os
import sys
import time
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "rule-chef-v2"))

import spacy
from spacy_labeling import annotate_sentence
from utilities.common import (
    collapse_bio,
    prepare_split,
    print_split_counts,
    save_predictions_conllu,
    evaluate_predictions,
    evaluate_overlap,
    labels_to_spans,
    write_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "manual_annotation", "hand_labelled.conllu"),
        help="Datafile that is used.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(THIS_DIR, "results", "SpacyLabeling"),
        help="Directory for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, dev_set, split_counts = prepare_split(args.data)
    print_split_counts(split_counts)

    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        raise SystemExit("spaCy model not found. Run: python -m spacy download de_core_news_lg")

    start = time.perf_counter()

    gold_labels = []
    pred_labels = []
    sentence_predictions = []
    gold_spans_all = []
    pred_spans_all = []

    for sent in dev_set:
        pred = annotate_sentence(sent, nlp)
        collapsed_gold = collapse_bio(sent["labels"])
        collapsed_pred = collapse_bio(pred)
        gold_labels.append(collapsed_gold)
        pred_labels.append(collapsed_pred)
        sentence_predictions.append(
            {"tokens": sent["token_forms"], "gold": collapsed_gold, "pred": collapsed_pred}
        )
        gold_spans_all.append(labels_to_spans(collapsed_gold, sent["token_forms"], sent["text"]))
        pred_spans_all.append(labels_to_spans(collapsed_pred, sent["token_forms"], sent["text"]))

    elapsed = time.perf_counter() - start
    report_text, report_dict = evaluate_predictions(gold_labels, pred_labels)
    overlap_metrics = evaluate_overlap(gold_spans_all, pred_spans_all)

    out_dir = Path(args.results_dir)
    write_metrics(out_dir, "SpacyLabeling", report_text, report_dict, elapsed, overlap_metrics)
    save_predictions_conllu(sentence_predictions, out_dir / "predictions.conllu")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
