#!/usr/bin/env python3
"""
Comprehensive comparison for final submission.

Runs multiple methods on the same stratified 80/20 split (seeded) and
produces comparable token-level metrics plus CoNLL-U prediction files.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import classification_report

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rule-chef-v2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "openai-ner"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "future_work", "baselines"))

from utils.stratified_split import SPLIT_SEED, stratified_split

from data_loader import _align_tokens, _tokens_to_spans
from inference import predict as rulechef_predict
from trainer import LABELS, train_all
from extractor import extract as openai_extract
from openai import OpenAI
from rule_based_ner import RuleBasedNER
from milestone2.milestone2 import SimpleRuleNER, TokenNB, build_token_label_pairs


LABEL_ORDER = ["B-LEG", "I-LEG", "B-MON", "I-MON", "B-ORG", "I-ORG", "O"]


def _reconstruct_text(tokens: List[Dict]) -> str:
    parts = []
    for tok in tokens:
        parts.append(tok["form"])
        if "SpaceAfter=No" not in tok.get("misc", ""):
            parts.append(" ")
    return "".join(parts).strip()


def load_conllu_full(path: str) -> List[Dict]:
    """
    Load CoNLL-U with tokens, labels, and sentence text.
    """
    sentences = []
    tokens: List[Dict] = []
    labels: List[str] = []
    sent_text = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    text = sent_text or _reconstruct_text(tokens)
                    token_forms = [t["form"] for t in tokens]
                    spans = _tokens_to_spans(list(zip(token_forms, labels)), text)
                    sentences.append(
                        {
                            "text": text,
                            "tokens": tokens,
                            "token_forms": token_forms,
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

            fields = line.split("\t")
            if len(fields) < 11 or "-" in fields[0]:
                continue

            token = {
                "form": fields[1],
                "upos": fields[3],
                "misc": fields[9],
            }
            tokens.append(token)
            labels.append(fields[10].strip())

    if tokens:
        text = sent_text or _reconstruct_text(tokens)
        token_forms = [t["form"] for t in tokens]
        spans = _tokens_to_spans(list(zip(token_forms, labels)), text)
        sentences.append(
            {
                "text": text,
                "tokens": tokens,
                "token_forms": token_forms,
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
            for j, (token, label) in enumerate(
                zip(sent_pred["tokens"], sent_pred["pred"])
            ):
                f.write(f"{j+1}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\t{label}\n")
            f.write("\n")


def evaluate_predictions(
    name: str, dev_set: List[Dict], pred_labels: List[List[str]]
) -> Dict:
    y_true, y_pred = [], []
    sentence_predictions = []

    for sent, pred in zip(dev_set, pred_labels):
        gold = sent["labels"]
        if len(pred) != len(gold):
            raise ValueError(f"{name}: length mismatch between gold and prediction")
        y_true.extend(gold)
        y_pred.extend(pred)
        sentence_predictions.append(
            {"tokens": sent["token_forms"], "gold": gold, "pred": pred}
        )

    report = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, digits=3, zero_division=0
    )
    return {"report": report, "sent_preds": sentence_predictions}


def _write_timing(f, timings: Dict[str, float]):
    f.write("Timing (seconds)\n")
    f.write("-" * 80 + "\n")
    for name in sorted(timings.keys()):
        f.write(f"{name}: {timings[name]:.2f}\n")
    f.write("\n")


def _log_step(message: str):
    print(f"[final] {message}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "manual_annotation2", "my_labels.conllu"),
        help="Single CoNLL-U file, will be split 80/20 into train/dev.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(THIS_DIR, "results"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--run-openai",
        action="store_true",
        help="Include OpenAI NER (requires API key).",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="OpenAI model name for direct API NER.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_sents = load_conllu_full(args.data)
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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    timings: Dict[str, float] = {}

    # Milestone 2 baselines
    _log_step("Running milestone2 baselines...")
    train_pairs = build_token_label_pairs(
        [{"tokens": s["token_forms"], "labels": s["labels"]} for s in train_set]
    )
    dev_tokens = [s["token_forms"] for s in dev_set]

    rule_model = SimpleRuleNER()
    start = time.perf_counter()
    rule_preds = [rule_model.predict_sentence(toks) for toks in dev_tokens]
    timings["SimpleRuleNER"] = time.perf_counter() - start
    all_results["SimpleRuleNER"] = evaluate_predictions(
        "SimpleRuleNER", dev_set, rule_preds
    )

    nb_imb = TokenNB(use_uniform_priors=False)
    start = time.perf_counter()
    nb_imb.count_tokens(train_pairs)
    nb_imb.calculate_weights()
    nb_imb_preds = [nb_imb.predict_sentence(toks) for toks in dev_tokens]
    timings["TokenNB_data_priors"] = time.perf_counter() - start
    all_results["TokenNB_data_priors"] = evaluate_predictions(
        "TokenNB (data priors)", dev_set, nb_imb_preds
    )

    nb_bal = TokenNB(use_uniform_priors=True)
    start = time.perf_counter()
    nb_bal.count_tokens(train_pairs)
    nb_bal.calculate_weights()
    nb_bal_preds = [nb_bal.predict_sentence(toks) for toks in dev_tokens]
    timings["TokenNB_uniform_priors"] = time.perf_counter() - start
    all_results["TokenNB_uniform_priors"] = evaluate_predictions(
        "TokenNB (uniform priors)", dev_set, nb_bal_preds
    )

    # Future work rule-based NER
    _log_step("Running future_work rule-based NER...")
    fw_model = RuleBasedNER()
    start = time.perf_counter()
    fw_preds = [fw_model.predict(s["tokens"]) for s in dev_set]
    timings["FutureWork_RuleBasedNER"] = time.perf_counter() - start
    all_results["FutureWork_RuleBasedNER"] = evaluate_predictions(
        "FutureWork RuleBasedNER", dev_set, fw_preds
    )

    # RuleChef
    _log_step("Training + running RuleChef...")
    data_by_label = build_data_by_label(train_set)
    storage_path = os.path.join(PROJECT_ROOT, "rule-chef-v2", "rulechef_v2_data")
    start = time.perf_counter()
    chefs = train_all(data_by_label, storage_path=storage_path)
    rulechef_preds = []
    for sent in dev_set:
        spans = rulechef_predict(sent["text"], chefs)
        rulechef_preds.append(spans_to_bio(spans, sent["token_forms"], sent["text"]))
    timings["RuleChef"] = time.perf_counter() - start
    all_results["RuleChef"] = evaluate_predictions("RuleChef", dev_set, rulechef_preds)

    # OpenAI NER (optional)
    if args.run_openai:
        _log_step("Running OpenAI NER...")
        client = OpenAI()
        start = time.perf_counter()
        openai_preds = []
        for sent in dev_set:
            spans = openai_extract(sent["text"], client, model=args.openai_model)
            openai_preds.append(spans_to_bio(spans, sent["token_forms"], sent["text"]))
        timings["OpenAI_NER"] = time.perf_counter() - start
        all_results["OpenAI_NER"] = evaluate_predictions(
            "OpenAI NER", dev_set, openai_preds
        )

    # Save predictions and metrics summary
    metrics_path = results_dir / "metrics_summary.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUBMISSION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        _write_timing(f, timings)
        for method_name, results in all_results.items():
            f.write(f"=== {method_name} ===\n")
            f.write(results["report"] + "\n\n")

    # Per-model metrics files
    per_model_dir = results_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)
    for method_name, results in all_results.items():
        safe_name = method_name.replace(" ", "_")
        out_path = per_model_dir / f"{safe_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{method_name} METRICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write("Timing (seconds)\n")
            f.write("-" * 80 + "\n")
            if method_name in timings:
                f.write(f"{method_name}: {timings[method_name]:.2f}\n\n")
            f.write(results["report"] + "\n")

    preds_dir = results_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for method_name, results in all_results.items():
        out_path = preds_dir / f"{method_name}.conllu"
        save_predictions_conllu(results["sent_preds"], out_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_dir}")


if __name__ == "__main__":
    main()
