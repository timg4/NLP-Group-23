#!/usr/bin/env python3
"""
Run German BERT token classification on stratified 80/20 split.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import warnings
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "rule-chef-v2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "final_submission", "utilities"))

from common import (
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
        default=os.path.join(THIS_DIR, "results", "BERT"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--model",
        default="bert-base-german-cased",
        help="Hugging Face model name.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def build_label_maps(sentences) -> List[str]:
    labels = set()
    for sent in sentences:
        for lab in sent["labels"]:
            labels.add(lab)
    labels = sorted(labels)
    return labels


def tokenize_and_align(sentences, tokenizer, label2id):
    tokens = [s["token_forms"] for s in sentences]
    labels = [s["labels"] for s in sentences]

    encodings = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=False,
    )

    aligned_labels = []
    for i, word_labels in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            else:
                lab = word_labels[word_id]
                if lab.startswith("B-"):
                    lab = "I-" + lab[2:]
                label_ids.append(label2id.get(lab, label2id[word_labels[word_id]]))
            prev_word_id = word_id
        aligned_labels.append(label_ids)

    encodings["labels"] = aligned_labels
    return encodings


def decode_predictions(sentences, encodings, logits, id2label):
    pred_ids = logits.argmax(-1)
    all_preds = []
    for i, sent in enumerate(sentences):
        word_ids = encodings.word_ids(batch_index=i)
        pred_labels = []
        prev_word_id = None
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            label_id = int(pred_ids[i][token_idx])
            pred_labels.append(id2label[label_id])
            prev_word_id = word_id
        all_preds.append(pred_labels)
    return all_preds


def main():
    args = parse_args()
    train_set, dev_set, split_counts = prepare_split(args.data)
    print_split_counts(split_counts)

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*resume_download.*|.*_register_pytree_node.*|.*dispatch_batches.*",
    )

    labels = build_label_maps(train_set + dev_set)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_enc = tokenize_and_align(train_set, tokenizer, label2id)
    dev_enc = tokenize_and_align(dev_set, tokenizer, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path(args.results_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(train_enc),
        eval_dataset=TokenDataset(dev_enc),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    preds = trainer.predict(TokenDataset(dev_enc))
    pred_labels_bio = decode_predictions(dev_set, dev_enc, preds.predictions, id2label)

    gold_labels = []
    pred_labels = []
    sentence_predictions = []
    gold_spans_all = []
    pred_spans_all = []

    for sent, pred in zip(dev_set, pred_labels_bio):
        collapsed_gold = collapse_bio(sent["labels"])
        collapsed_pred = collapse_bio(pred)
        gold_labels.append(collapsed_gold)
        pred_labels.append(collapsed_pred)
        sentence_predictions.append(
            {"tokens": sent["token_forms"], "gold": collapsed_gold, "pred": collapsed_pred}
        )
        gold_spans_all.append(labels_to_spans(collapsed_gold, sent["token_forms"], sent["text"]))
        pred_spans_all.append(labels_to_spans(collapsed_pred, sent["token_forms"], sent["text"]))

    report_text, report_dict = evaluate_predictions(gold_labels, pred_labels)
    overlap_metrics = evaluate_overlap(gold_spans_all, pred_spans_all)

    out_dir = Path(args.results_dir)
    write_metrics(out_dir, "BERT", report_text, report_dict, elapsed, overlap_metrics)
    save_predictions_conllu(sentence_predictions, out_dir / "predictions.conllu")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
