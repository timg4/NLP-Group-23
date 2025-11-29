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

    for sent in sentences:
        gold = sent["labels"]
        pred = model.predict_sentence(sent["tokens"])
        if len(pred) != len(gold):
            raise ValueError("Length mismatch between gold and prediction")
        y_true.extend(gold)
        y_pred.extend(pred)

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=3))


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

    # 1) Rule-based
    rule_model = SimpleRuleNER()
    evaluate_model(rule_model, dev_set, name="SimpleRuleNER")

    # 2) Naive Bayes with label priors from data (biased to 'O')
    nb_imb = TokenNB(use_uniform_priors=False)
    nb_imb.count_tokens(train_pairs)
    nb_imb.calculate_weights()
    evaluate_model(nb_imb, dev_set, name="TokenNB (data priors)")

    # 3) Naive Bayes with uniform label priors (handles imbalance a bit)
    nb_bal = TokenNB(use_uniform_priors=True)
    nb_bal.count_tokens(train_pairs)
    nb_bal.calculate_weights()
    evaluate_model(nb_bal, dev_set, name="TokenNB (uniform priors)")


if __name__ == "__main__":
    main()
