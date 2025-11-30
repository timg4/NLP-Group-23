#!/usr/bin/env python
"""
Train/Evaluate Rule-Based NER Baseline
-------------------------------------------------------------
Evaluate rule-based NER on dev/test set

Usage:  python baselines/train_rule_based.py \
        --dev data/processed/splits/dev.conllu \
        --output results/rule_based
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json

from rule_based_ner import RuleBasedNER, load_conllu


def save_predictions(sentences, predictions, output_path: Path):
    """Save predictions in CoNLL-U format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for tokens, pred_tags in zip(sentences, predictions):
            for token, tag in zip(tokens, pred_tags):
                line = '\t'.join([
                    token['id'],
                    token['form'],
                    token['lemma'],
                    token['upos'],
                    token['xpos'],
                    token['feats'],
                    token['head'],
                    token['deprel'],
                    token['deps'],
                    token['misc'],
                    tag
                ])
                f.write(line + '\n')
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rule-Based NER")
    parser.add_argument("--dev", required=True, help="Dev set CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output directory for results")
    args = parser.parse_args()

    dev_path = Path(args.dev)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Rule-Based NER Evaluation")
    print("="*60)

    # Load dev set
    print(f"\nLoading dev set: {dev_path}")
    sentences = load_conllu(str(dev_path))
    print(f"Loaded {len(sentences)} sentences")

    # Initialize rule-based NER
    print("\nInitializing rule-based NER...")
    ner = RuleBasedNER()

    # Predict on all sentences
    print("\nPredicting NER tags...")
    y_true = []
    y_pred = []

    for tokens, gold_tags in tqdm(sentences, desc="Sentences"):
        pred_tags = ner.predict(tokens)
        y_true.append(gold_tags)
        y_pred.append(pred_tags)

    # Save predictions
    pred_output = output_dir / "predictions.conllu"
    print(f"\nSaving predictions to: {pred_output}")
    save_predictions([tokens for tokens, _ in sentences], y_pred, pred_output)

    # Calculate and save metrics (will be done by evaluation script)
    print(f"\nPredictions saved. Run evaluation script to calculate metrics:")
    print(f"  python evaluation/metrics.py --gold {dev_path} --pred {pred_output}")

    # Save metadata
    metadata = {
        "model": "rule_based",
        "dev_file": str(dev_path),
        "num_sentences": len(sentences),
        "num_tokens": sum(len(tokens) for tokens, _ in sentences)
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
