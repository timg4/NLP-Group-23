#!/usr/bin/env python
"""
Train CRF-Based NER Baseline
-------------------------------------------------------------
Train and evaluate CRF NER on train and dev sets

Usage:  python baselines/train_crf.py \
        --train data/processed/splits/train.conllu \
        --dev data/processed/splits/dev.conllu \
        --output results/crf
"""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import json

from crf_ner import CRFNER, load_conllu


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
    parser = argparse.ArgumentParser(description="Train and Evaluate CRF NER")
    parser.add_argument("--train", required=True, help="Train set CoNLL-U file")
    parser.add_argument("--dev", required=True, help="Dev set CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--c1", type=float, default=0.1, help="L1 regularization")
    parser.add_argument("--c2", type=float, default=0.1, help="L2 regularization")
    parser.add_argument("--max_iterations", type=int, default=100, help="Max training iterations")
    args = parser.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CRF-Based NER Training and Evaluation")
    print("="*60)

    # Load train set
    print(f"\nLoading train set: {train_path}")
    train_sentences = load_conllu(str(train_path))
    print(f"Loaded {len(train_sentences)} training sentences")

    # Load dev set
    print(f"\nLoading dev set: {dev_path}")
    dev_sentences = load_conllu(str(dev_path))
    print(f"Loaded {len(dev_sentences)} dev sentences")

    # Initialize CRF
    print("\nInitializing CRF model...")
    print(f"  L1 regularization (c1): {args.c1}")
    print(f"  L2 regularization (c2): {args.c2}")
    print(f"  Max iterations: {args.max_iterations}")

    crf = CRFNER(c1=args.c1, c2=args.c2, max_iterations=args.max_iterations)

    # Extract features
    print("\nExtracting features for training...")
    X_train = []
    y_train = []

    for tokens, tags in tqdm(train_sentences, desc="Train"):
        X_train.append(crf.sentence_features(tokens))
        y_train.append(tags)

    print(f"\nExtracting features for dev...")
    X_dev = []
    y_dev = []

    for tokens, tags in tqdm(dev_sentences, desc="Dev"):
        X_dev.append(crf.sentence_features(tokens))
        y_dev.append(tags)

    # Train model
    print("\nTraining CRF model...")
    print("This may take a few minutes...")
    crf.fit(X_train, y_train)
    print("Training completed!")

    # Predict on dev set
    print("\nPredicting on dev set...")
    y_pred = crf.predict(X_dev)

    # Save model
    model_path = output_dir / "model.pkl"
    print(f"\nSaving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)

    # Save predictions
    pred_output = output_dir / "predictions.conllu"
    print(f"Saving predictions to: {pred_output}")
    save_predictions([tokens for tokens, _ in dev_sentences], y_pred, pred_output)

    # Analyze feature weights
    print("\nAnalyzing feature weights...")
    feature_weights = crf.get_feature_weights(top_n=15)

    weights_output = output_dir / "feature_weights.txt"
    with open(weights_output, 'w', encoding='utf-8') as f:
        f.write("Top Features by Entity Type\n")
        f.write("="*60 + "\n\n")

        for tag, features in sorted(feature_weights.items()):
            if tag != 'O':
                f.write(f"\n{tag}:\n")
                f.write("-"*40 + "\n")
                for feat, weight in features:
                    f.write(f"  {weight:8.4f}  {feat}\n")

    print(f"Feature weights saved to: {weights_output}")

    # Save metadata
    metadata = {
        "model": "crf",
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "num_train_sentences": len(train_sentences),
        "num_dev_sentences": len(dev_sentences),
        "c1": args.c1,
        "c2": args.c2,
        "max_iterations": args.max_iterations
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPredictions saved. Run evaluation script to calculate metrics:")
    print(f"  python evaluation/metrics.py --gold {dev_path} --pred {pred_output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
