#!/usr/bin/env python
"""
Milestone 2 baselines for German financial NER.

Models:
1) SimpleRuleNER (rule-based, no training)
2) TokenNB (Naive Bayes over tokens, label priors from data -> biased to 'O')
3) TokenNB (Naive Bayes over tokens, uniform priors -> compensates class imbalance)

Usage examples:

    # Single file, 80/20 split:
    python milestone2_baselines.py --data hand_labels.conllu

"""

import argparse
import re
from collections import Counter, defaultdict
from math import log
from typing import List, Dict

from sklearn.metrics import classification_report
import os
from pathlib import Path
from collections import defaultdict


#load data

def load_conllu(path: str) -> List[Dict[str, List[str]]]:
    """
    Load a CoNLL-U file where the last column holds BIO labels.

    Returns: list of dicts: {"tokens": [...], "labels": [...]}
    """
    sentences = []
    tokens, labels = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            tok = parts[1]
            lab = parts[-1]
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        sentences.append({"tokens": tokens, "labels": labels})

    return sentences


def build_token_label_pairs(sentences: List[Dict[str, List[str]]]):
    """
    Flatten sentence-level structure into (token, label) pairs.
    """
    pairs = []
    for sent in sentences:
        for tok, lab in zip(sent["tokens"], sent["labels"]):
            pairs.append((tok, lab))
    return pairs


#baseline: Rule Based NER - simple rules for all different tokens 

class SimpleRuleNER:
    """
    Very simple hand-written rules for LEG, MON, ORG.
    """

    import re

class SimpleRuleNER:
    def __init__(self):
        # patterns for monetary stuff
        # e.g. 3,00  21.396  2,50%  0%  2,0
        self.mon_number_re = re.compile(
            r"^\d{1,3}(\.\d{3})*(,\d+)?$"    
            r"|^\d+(,\d+)?%$"                  
        )
        self.currency_tokens = {"€", "eur", "eur.", "euro", "euro."}

        # legal patterns
        self.leg_starters = {"§", "Artikel", "Artikel.", "Art.", "Art"}
        self.leg_follow = {
            "Absatz", "Abs.", "Satz", "Nr.", "Nr", "Nummer",
            "(", ")", "-", "WpHG"
        }

        # organization patterns, banks etc
        self.org_keywords = {
            "bank", "bundesbank", "sparkasse", "genossenschaftsbank"
        }
        self.org_suffixes = {
            "ag", "gmbh", "kg", "kgaa", "se", "plc", "ltd", "llc", "inc.", "sarl"
        }

    # ------- helpers --------

    def _is_numeric_like(self, tok: str) -> bool:
        return any(ch.isdigit() for ch in tok)

    def predict_sentence(self, tokens):
        n = len(tokens)
        labels = ["O"] * n
        lower = [t.lower() for t in tokens]

        # rules for leagal stuff
        i = 0
        while i < n:
            t = tokens[i]
            tl = lower[i]

            # Pattern could be § 123 
            if t == "§":
                labels[i] = "B-LEG"
                j = i + 1
                # then we take a short window of tokens after 
                while j < n and tokens[j] not in {".", ";"} and j < i + 8:
                    labels[j] = "I-LEG"
                    j += 1
                i = j
                continue

            # Patrern is Artikel ...
            if t in {"Artikel", "Artikel.", "Art.", "Art"}:
                labels[i] = "B-LEG"
                j = i + 1
                #continues with numbers or legal words
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    if self._is_numeric_like(tj) or t in self.leg_follow or tlj in {lf.lower() for lf in self.leg_follow}:
                        labels[j] = "I-LEG"
                        j += 1
                    else:
                        break
                i = j
                continue

            i += 1

        # rulkes for monetary stuff
        for i, t in enumerate(tokens):
            if labels[i] != "O":
                #don't overwrite LEG
                continue
            tl = lower[i]
            prev_tok = lower[i-1] if i > 0 else ""
            next_tok = lower[i+1] if i + 1 < n else ""

            #Pattern could b numeric with thousand/decimal separators or "%"
            if self.mon_number_re.match(t) or t.endswith("%"):
                labels[i] = "B-MON"
                # attach directly adjacent % or currency tokens
                j = i + 1
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    if (
                        tj.endswith("%")
                        or tlj in self.currency_tokens
                        or self.mon_number_re.match(tj)
                    ):
                        if labels[j] == "O":
                            labels[j] = "I-MON"
                        j += 1
                    else:
                        break
                continue

            # Pattern could be currency word with numeric before it
            if tl in self.currency_tokens and i > 0 and self._is_numeric_like(tokens[i-1]):
                if labels[i-1] == "O":
                    labels[i-1] = "B-MON"
                labels[i] = "I-MON"

        # rules for organizational stuff
        for i, t in enumerate(tokens):
            if labels[i] != "O":
                continue
            tl = lower[i]

            # Pattern could be lexicon hits like BANK
            if tl in self.org_keywords:
                labels[i] = "B-ORG"
                j = i + 1
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    #extend through typical name pieces
                    if (
                        tj[0].isupper()
                        or tlj in self.org_keywords
                        or tlj in self.org_suffixes
                        or tj in {"-", "–", "&", "’s", "'s", ",", "."}
                    ):
                        if labels[j] == "O":
                            labels[j] = "I-ORG"
                        j += 1
                    else:
                        break
                continue

            # Pattern could be legal form suffix like "AG", "GmbH" etc.
            if tl in self.org_suffixes:
                #go backwards to find start of name chunk
                start = i
                while start - 1 >= 0 and tokens[start - 1][0].isupper() and labels[start - 1] == "O":
                    start -= 1
                labels[start] = "B-ORG"
                for j in range(start + 1, i + 1):
                    if labels[j] == "O":
                        labels[j] = "I-ORG"

        return labels


# also implement a Naive Bayes classifier over tokens

class TokenNB:
    """
    Naive Bayes classifier over single tokens.
    We use Laplace smoothing for P(token | label).
    """

    def __init__(self, use_uniform_priors: bool = False):
        self.use_uniform_priors = use_uniform_priors
        self.labels = set()
        self.word_count = Counter()
        self.count_by_label = defaultdict(Counter)
        self.label_count = Counter()
        self.trained = False

    def count_tokens(self, token_label_pairs):
        for word, label in token_label_pairs:
            self.labels.add(label)
            self.word_count[word] += 1
            self.count_by_label[label][word] += 1
            self.label_count[label] += 1

    def calculate_weights(self):
        V = len(self.word_count)
        total_labels = sum(self.label_count.values())

        #log P(word | label)
        self.weights = {}
        for word in self.word_count:
            self.weights[word] = {}
            for label in self.labels:
                num = self.count_by_label[label][word] + 1  # Laplace
                denom = self.label_count[label] + V
                self.weights[word][label] = log(num / denom)

        #log P(label)
        if self.use_uniform_priors:
            self.label_priors = {label: 0.0 for label in self.labels}
        else:
            self.label_priors = {}
            for label in self.labels:
                prior = (self.label_count[label] + 1) / (total_labels + len(self.labels))
                self.label_priors[label] = log(prior)

        self.trained = True

    def predict_sentence(self, tokens: List[str]) -> List[str]:
        if not self.trained:
            raise RuntimeError("Call calculate_weights() first")

        preds = []
        for word in tokens:
            if word not in self.weights:
                # unseen: default to 'O'
                preds.append("O")
                continue

            best_label = None
            best_score = float("-inf")
            for label in self.labels:
                score = self.label_priors[label] + self.weights[word][label]
                if score > best_score:
                    best_score = score
                    best_label = label
            preds.append(best_label)
        return preds


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

LABEL_ORDER = ["B-LEG", "I-LEG", "B-MON", "I-MON", "B-ORG", "I-ORG", "O"]


def evaluate_model(model, sentences, name=""):
    y_true, y_pred = [], []
    sentence_predictions = []

    for sent in sentences:
        gold = sent["labels"]
        pred = model.predict_sentence(sent["tokens"])
        if len(pred) != len(gold):
            raise ValueError("Length mismatch between gold and prediction")
        y_true.extend(gold)
        y_pred.extend(pred)
        sentence_predictions.append({
            "tokens": sent["tokens"],
            "gold": gold,
            "pred": pred
        })

    print(f"\n=== {name} ===")
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=3)
    print(report)

    return report, sentence_predictions, y_true, y_pred


# ----------------------------------------------------------------------
# Output Generation
# ----------------------------------------------------------------------

def save_predictions_conllu(sentence_predictions, output_path, sent_id_offset=0):
    """Save predictions in CoNLL-U format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sent_pred in enumerate(sentence_predictions):
            f.write(f"# sent_id = {sent_id_offset + i + 1}\n")
            for j, (token, label) in enumerate(zip(sent_pred["tokens"], sent_pred["pred"])):
                # CoNLL-U format: ID, FORM, ... (10 columns), NER_TAG (11th column)
                f.write(f"{j+1}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\t{label}\n")
            f.write("\n")


def analyze_errors(sentence_predictions, y_true, y_pred):
    """Generate error analysis by category."""
    from sklearn.metrics import confusion_matrix

    analysis = []
    analysis.append("=" * 80)
    analysis.append("ERROR ANALYSIS")
    analysis.append("=" * 80)
    analysis.append("")

    # Count errors by entity type
    entity_errors = defaultdict(lambda: {"FP": 0, "FN": 0, "boundary": 0})

    for sent_pred in sentence_predictions:
        gold = sent_pred["gold"]
        pred = sent_pred["pred"]

        for i, (g, p) in enumerate(zip(gold, pred)):
            if g != p:
                # False positive: predicted entity when gold is O
                if g == "O" and p != "O":
                    entity_type = p.split("-")[1] if "-" in p else p
                    entity_errors[entity_type]["FP"] += 1
                # False negative: missed entity
                elif g != "O" and p == "O":
                    entity_type = g.split("-")[1] if "-" in g else g
                    entity_errors[entity_type]["FN"] += 1
                # Boundary errors (B/I confusion or wrong entity type)
                elif g != "O" and p != "O":
                    entity_type = g.split("-")[1] if "-" in g else g
                    entity_errors[entity_type]["boundary"] += 1

    analysis.append("Errors by Entity Type:")
    analysis.append("-" * 80)
    for ent_type in ["ORG", "MON", "LEG"]:
        errs = entity_errors[ent_type]
        analysis.append(f"{ent_type}:")
        analysis.append(f"  False Positives (predicted {ent_type}, actually O): {errs['FP']}")
        analysis.append(f"  False Negatives (missed {ent_type}): {errs['FN']}")
        analysis.append(f"  Boundary Errors (B/I confusion or type mismatch): {errs['boundary']}")
        analysis.append("")

    return "\n".join(analysis)


def extract_example_sentences(sentence_predictions, num_examples=5):
    """Extract representative example sentences for qualitative analysis."""
    examples = []
    examples.append("=" * 80)
    examples.append("EXAMPLE SENTENCES")
    examples.append("=" * 80)
    examples.append("")

    # Find sentences with different error patterns
    selected = []

    # 1. Sentences with all correct predictions
    for sent_pred in sentence_predictions:
        if sent_pred["gold"] == sent_pred["pred"] and any(l != "O" for l in sent_pred["gold"]):
            selected.append(("CORRECT", sent_pred))
            break

    # 2. Sentences with errors
    error_sents = []
    for sent_pred in sentence_predictions:
        if sent_pred["gold"] != sent_pred["pred"]:
            error_sents.append(sent_pred)

    # Take diverse error examples
    if error_sents:
        selected.extend([("ERROR", s) for s in error_sents[:num_examples-1]])

    for label, sent_pred in selected[:num_examples]:
        tokens = sent_pred["tokens"]
        gold = sent_pred["gold"]
        pred = sent_pred["pred"]

        examples.append(f"[{label}] Sentence: {' '.join(tokens)}")
        examples.append("")
        examples.append(f"{'Token':<20} {'Gold':<15} {'Predicted':<15}")
        examples.append("-" * 50)

        for token, g, p in zip(tokens, gold, pred):
            marker = "✓" if g == p else "✗"
            examples.append(f"{token:<20} {g:<15} {p:<15} {marker}")

        examples.append("")
        examples.append("=" * 80)
        examples.append("")

    return "\n".join(examples)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Single CoNLL-U file, will be split 80/20 into train/dev.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # always: one file, 80/20 split
    all_sents = load_conllu(args.data)
    split_idx = int(0.8 * len(all_sents))
    train_set = all_sents[:split_idx]
    dev_set = all_sents[split_idx:]

    print(f"Total sentences: {len(all_sents)}")
    print(f"Train sentences: {len(train_set)}")
    print(f"Dev sentences:   {len(dev_set)}")

    train_pairs = build_token_label_pairs(train_set)

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Store all results for combined analysis
    all_results = {}

    # 1) Rule-based
    print("\n" + "=" * 80)
    print("EVALUATING: SimpleRuleNER")
    print("=" * 80)
    rule_model = SimpleRuleNER()
    report_rule, sent_preds_rule, y_true_rule, y_pred_rule = evaluate_model(
        rule_model, dev_set, name="SimpleRuleNER"
    )
    all_results["SimpleRuleNER"] = {
        "report": report_rule,
        "sent_preds": sent_preds_rule,
        "y_true": y_true_rule,
        "y_pred": y_pred_rule
    }

    # 2) Naive Bayes with label priors from data (biased to 'O')
    print("\n" + "=" * 80)
    print("EVALUATING: TokenNB (data priors)")
    print("=" * 80)
    nb_imb = TokenNB(use_uniform_priors=False)
    nb_imb.count_tokens(train_pairs)
    nb_imb.calculate_weights()
    report_nb_data, sent_preds_nb_data, y_true_nb_data, y_pred_nb_data = evaluate_model(
        nb_imb, dev_set, name="TokenNB (data priors)"
    )
    all_results["TokenNB_data_priors"] = {
        "report": report_nb_data,
        "sent_preds": sent_preds_nb_data,
        "y_true": y_true_nb_data,
        "y_pred": y_pred_nb_data
    }

    # 3) Naive Bayes with uniform label priors (handles imbalance a bit)
    print("\n" + "=" * 80)
    print("EVALUATING: TokenNB (uniform priors)")
    print("=" * 80)
    nb_bal = TokenNB(use_uniform_priors=True)
    nb_bal.count_tokens(train_pairs)
    nb_bal.calculate_weights()
    report_nb_uniform, sent_preds_nb_uniform, y_true_nb_uniform, y_pred_nb_uniform = evaluate_model(
        nb_bal, dev_set, name="TokenNB (uniform priors)"
    )
    all_results["TokenNB_uniform_priors"] = {
        "report": report_nb_uniform,
        "sent_preds": sent_preds_nb_uniform,
        "y_true": y_true_nb_uniform,
        "y_pred": y_pred_nb_uniform
    }

    # Save predictions in CoNLL-U format
    print("\n" + "=" * 80)
    print("SAVING PREDICTIONS")
    print("=" * 80)

    save_predictions_conllu(
        sent_preds_rule,
        results_dir / "rule_based_predictions.conllu"
    )
    print(f"✓ Saved: {results_dir / 'rule_based_predictions.conllu'}")

    save_predictions_conllu(
        sent_preds_nb_data,
        results_dir / "nb_data_priors_predictions.conllu"
    )
    print(f"✓ Saved: {results_dir / 'nb_data_priors_predictions.conllu'}")

    save_predictions_conllu(
        sent_preds_nb_uniform,
        results_dir / "nb_uniform_priors_predictions.conllu"
    )
    print(f"✓ Saved: {results_dir / 'nb_uniform_priors_predictions.conllu'}")

    # Save metrics summary
    metrics_path = results_dir / "metrics_summary.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MILESTONE 2 BASELINE METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for method_name, results in all_results.items():
            f.write(f"=== {method_name} ===\n")
            f.write(results["report"] + "\n\n")

    print(f"✓ Saved: {metrics_path}")

    # Generate and save error analysis for each method
    error_analysis_path = results_dir / "error_analysis.txt"
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        for method_name, results in all_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"ERROR ANALYSIS: {method_name}\n")
            f.write(f"{'=' * 80}\n\n")
            error_text = analyze_errors(
                results["sent_preds"],
                results["y_true"],
                results["y_pred"]
            )
            f.write(error_text + "\n\n")

    print(f"✓ Saved: {error_analysis_path}")

    # Extract and save example sentences
    examples_path = results_dir / "example_sentences.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXAMPLE SENTENCES FOR QUALITATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for method_name, results in all_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"METHOD: {method_name}\n")
            f.write(f"{'=' * 80}\n\n")
            examples_text = extract_example_sentences(results["sent_preds"], num_examples=3)
            f.write(examples_text + "\n\n")

    print(f"✓ Saved: {examples_path}")

    print("\n" + "=" * 80)
    print("ALL OUTPUTS SAVED TO: milestone2/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
