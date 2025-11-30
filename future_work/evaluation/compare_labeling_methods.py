#!/usr/bin/env python
"""
Compare Labeling Methods
-------------------------------------------------------------
Compare different NER labeling methods against manual gold standard:
1. Manual (gold standard)
2. ChatGPT
3. Project (spaCy + Regex)

Calculates metrics, agreement, and provides detailed error analysis.

Usage:  python evaluation/compare_labeling_methods.py --gold data/manual_annotation/sample_sentences_labeled.conllu --chatgpt results/labeling_comparison/chatgpt_predictions.tsv --project results/labeling_comparison/project_predictions.conllu --output results/labeling_comparison
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def load_gold_from_tsv(file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Load manually labeled data from TSV"""
    df = pd.read_csv(file_path, sep='\t', keep_default_na=False, quoting=3)  # QUOTE_NONE

    # Group by sentence_id
    sentences_tokens = []
    sentences_tags = []

    for sent_id, group in df.groupby('sentence_id'):
        # Filter out empty rows (sentence separators)
        group = group[group['token'].str.strip() != '']

        tokens = group['token'].tolist()
        tags = group['ner_tag'].tolist()

        if tokens and tags:
            sentences_tokens.append(tokens)
            sentences_tags.append(tags)

    print(f"Loaded {len(sentences_tags)} sentences from gold standard")
    return sentences_tokens, sentences_tags


def load_predictions_from_conllu(file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Load predictions from CoNLL-U file"""
    sentences_tokens = []
    sentences_tags = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:  # Sentence boundary
                if current_tokens:
                    sentences_tokens.append(current_tokens)
                    sentences_tags.append(current_tags)
                current_tokens = []
                current_tags = []

            elif not line.startswith('#'):
                fields = line.split('\t')
                if len(fields) >= 11 and '-' not in fields[0]:
                    current_tokens.append(fields[1])  # form
                    current_tags.append(fields[10].strip())  # NER tag

    if current_tokens:
        sentences_tokens.append(current_tokens)
        sentences_tags.append(current_tags)

    print(f"Loaded {len(sentences_tags)} sentences from predictions")
    return sentences_tokens, sentences_tags


def load_predictions_from_tsv(file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Load predictions from TSV file (same format as gold standard TSV)"""
    df = pd.read_csv(file_path, sep='\t', keep_default_na=False, quoting=3)  # QUOTE_NONE

    # Group by sentence_id (or sent_id)
    sent_id_col = 'sentence_id' if 'sentence_id' in df.columns else 'sent_id'
    token_col = 'token'
    tag_col = 'ner_tag'

    sentences_tokens = []
    sentences_tags = []

    for sent_id, group in df.groupby(sent_id_col):
        # Filter out empty rows (sentence separators)
        group = group[group[token_col].astype(str).str.strip() != '']

        tokens = group[token_col].tolist()
        tags = group[tag_col].tolist()

        if tokens and tags:
            sentences_tokens.append(tokens)
            sentences_tags.append(tags)

    print(f"Loaded {len(sentences_tags)} sentences from predictions")
    return sentences_tokens, sentences_tags


def validate_alignment(gold_tokens: List[List[str]], pred_tokens: List[List[str]], method_name: str):
    """Validate that gold and predicted tokens match"""
    if len(gold_tokens) != len(pred_tokens):
        print(f"WARNING: {method_name} - Sentence count mismatch!")
        print(f"  Gold: {len(gold_tokens)}, Predicted: {len(pred_tokens)}")
        return False

    for i, (gold_sent, pred_sent) in enumerate(zip(gold_tokens, pred_tokens)):
        if len(gold_sent) != len(pred_sent):
            print(f"WARNING: {method_name} - Sentence {i+1} token count mismatch!")
            print(f"  Gold: {len(gold_sent)}, Predicted: {len(pred_sent)}")
            return False

        # Check if tokens match (case-insensitive to be forgiving)
        for j, (gold_tok, pred_tok) in enumerate(zip(gold_sent, pred_sent)):
            if gold_tok.lower() != pred_tok.lower():
                print(f"WARNING: {method_name} - Sentence {i+1}, Token {j+1} mismatch!")
                print(f"  Gold: '{gold_tok}', Predicted: '{pred_tok}'")
                return False

    return True


def calculate_metrics(y_true: List[List[str]], y_pred: List[List[str]], method_name: str) -> Dict:
    """Calculate evaluation metrics"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    metrics = {
        'Method': method_name,
        'Overall_Precision': precision,
        'Overall_Recall': recall,
        'Overall_F1': f1,
    }

    # Per-entity metrics
    for entity_type in ['ORG', 'MON', 'LEG']:
        if entity_type in report:
            metrics[f'{entity_type}_Precision'] = report[entity_type]['precision']
            metrics[f'{entity_type}_Recall'] = report[entity_type]['recall']
            metrics[f'{entity_type}_F1'] = report[entity_type]['f1-score']
        else:
            metrics[f'{entity_type}_Precision'] = 0.0
            metrics[f'{entity_type}_Recall'] = 0.0
            metrics[f'{entity_type}_F1'] = 0.0

    # Token-level accuracy
    total_tokens = sum(len(sent) for sent in y_true)
    correct_tokens = sum(
        1 for sent_true, sent_pred in zip(y_true, y_pred)
        for tag_true, tag_pred in zip(sent_true, sent_pred)
        if tag_true == tag_pred
    )
    metrics['Token_Accuracy'] = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return metrics




def create_comparison_table(results: List[Dict], output_dir: Path):
    """Create and save comparison table"""
    df = pd.DataFrame(results)

    # Reorder columns
    column_order = [
        'Method',
        'Overall_Precision', 'Overall_Recall', 'Overall_F1',
        'ORG_Precision', 'ORG_Recall', 'ORG_F1',
        'MON_Precision', 'MON_Recall', 'MON_F1',
        'LEG_Precision', 'LEG_Recall', 'LEG_F1',
        'Token_Accuracy'
    ]
    df = df[column_order]

    # Save as CSV
    csv_path = output_dir / 'labeling_comparison.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved comparison table to: {csv_path}")

    # Save as Markdown
    md_path = output_dir / 'labeling_comparison.md'
    df.to_markdown(md_path, index=False, floatfmt='.4f')
    print(f"Saved comparison table to: {md_path}")

    return df


def plot_f1_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing F1 scores"""
    methods = df['Method'].tolist()
    entity_types = ['Overall', 'ORG', 'MON', 'LEG']

    f1_scores = {
        'Overall': df['Overall_F1'].tolist(),
        'ORG': df['ORG_F1'].tolist(),
        'MON': df['MON_F1'].tolist(),
        'LEG': df['LEG_F1'].tolist()
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(methods))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (entity_type, color) in enumerate(zip(entity_types, colors)):
        scores = f1_scores[entity_type]
        offset = width * (i - 1.5)
        bars = ax.bar([xi + offset for xi in x], scores, width, label=entity_type, color=color)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Labeling Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Labeling Method Comparison - F1 Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plot_path = output_dir / 'f1_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved F1 comparison plot to: {plot_path}")
    plt.close()




def generate_detailed_report(results: List[Dict], output_path: Path):
    """Generate detailed comparison report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LABELING METHOD COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Summary
        f.write("EVALUATION SUMMARY\n")
        f.write("-" * 70 + "\n\n")

        for result in results:
            f.write(f"{result['Method']}:\n")
            f.write(f"  Overall F1:      {result['Overall_F1']:.4f}\n")
            f.write(f"  Token Accuracy:  {result['Token_Accuracy']:.4f}\n")
            f.write(f"  ORG F1:          {result['ORG_F1']:.4f}\n")
            f.write(f"  MON F1:          {result['MON_F1']:.4f}\n")
            f.write(f"  LEG F1:          {result['LEG_F1']:.4f}\n")
            f.write("\n")

        # Recommendations
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")

        # Find best method
        best_method = max(results, key=lambda x: x['Overall_F1'])
        f.write(f"Best Overall Performance: {best_method['Method']}\n")
        f.write(f"  F1 Score: {best_method['Overall_F1']:.4f}\n\n")

        # Per-entity best
        for entity in ['ORG', 'MON', 'LEG']:
            best = max(results, key=lambda x: x[f'{entity}_F1'])
            f.write(f"Best for {entity}: {best['Method']} (F1: {best[f'{entity}_F1']:.4f})\n")

    print(f"\nSaved detailed report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare labeling methods")
    parser.add_argument("--gold", required=True, help="Gold standard file (manual labels, TSV or CoNLL-U)")
    parser.add_argument("--chatgpt", required=False, help="ChatGPT predictions file (TSV or CoNLL-U, optional)")
    parser.add_argument("--project", required=True, help="Project predictions CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    chatgpt_path = Path(args.chatgpt) if args.chatgpt else None
    project_path = Path(args.project)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LABELING METHOD COMPARISON")
    print("=" * 70)

    # Load gold standard
    print("\nLoading gold standard (manual labels)...")
    # Auto-detect format based on file extension
    if gold_path.suffix.lower() == '.tsv':
        gold_tokens, gold_tags = load_gold_from_tsv(gold_path)
    else:
        gold_tokens, gold_tags = load_predictions_from_conllu(gold_path)

    # Load ChatGPT predictions (if provided)
    chatgpt_tokens, chatgpt_tags = None, None
    if chatgpt_path:
        print("\nLoading ChatGPT predictions...")
        # Auto-detect format based on file extension
        if chatgpt_path.suffix.lower() == '.tsv':
            chatgpt_tokens, chatgpt_tags = load_predictions_from_tsv(chatgpt_path)
        else:
            chatgpt_tokens, chatgpt_tags = load_predictions_from_conllu(chatgpt_path)

    # Load Project predictions
    print("\nLoading Project predictions...")
    project_tokens, project_tags = load_predictions_from_conllu(project_path)

    # Validate alignment
    print("\nValidating alignment...")
    if chatgpt_path and not validate_alignment(gold_tokens, chatgpt_tokens, "ChatGPT"):
        print("ERROR: ChatGPT predictions don't align with gold standard!")
        return
    if not validate_alignment(gold_tokens, project_tokens, "Project"):
        print("ERROR: Project predictions don't align with gold standard!")
        return

    print("All predictions aligned with gold standard")

    # Calculate metrics
    print("\nCalculating metrics...")
    results = []

    if chatgpt_tags:
        chatgpt_metrics = calculate_metrics(gold_tags, chatgpt_tags, "ChatGPT")
        results.append(chatgpt_metrics)
        print(f"  ChatGPT F1: {chatgpt_metrics['Overall_F1']:.4f}")

    project_metrics = calculate_metrics(gold_tags, project_tags, "Project (spaCy+Regex)")
    results.append(project_metrics)
    print(f"  Project F1: {project_metrics['Overall_F1']:.4f}")

    # Generate outputs
    print("\nGenerating comparison outputs...")

    # Comparison table
    df = create_comparison_table(results, output_dir)

    # Plots
    plot_f1_comparison(df, output_dir)

    # Detailed report
    generate_detailed_report(results, output_dir / 'detailed_report.txt')

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nComparison Table:")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
