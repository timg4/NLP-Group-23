#!/usr/bin/env python
"""
Sample Sentences for Manual Annotation
-------------------------------------------------------------
Randomly sample sentences from corpus for manual labeling.
Ensures diversity and likely entity presence.

Usage:  python data/sample_for_manual_annotation.py --input data/processed/fincorpus_processed.conllu --output data/manual_annotation --num_sentences 150
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter


class Sentence:
    """Represents a sentence from CoNLL-U file"""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.comments = []
        self.tokens = []

    def add_comment(self, line: str):
        self.comments.append(line)

    def add_token(self, fields: List[str]):
        """Add token from CoNLL-U line"""
        if len(fields) >= 10 and '-' not in fields[0]:
            token = {
                'id': fields[0],
                'form': fields[1],
                'lemma': fields[2],
                'upos': fields[3],
                'xpos': fields[4],
                'feats': fields[5],
                'head': fields[6],
                'deprel': fields[7],
                'deps': fields[8],
                'misc': fields[9],
            }
            self.tokens.append(token)

    def get_text(self) -> str:
        """Reconstruct sentence text"""
        text = ""
        for token in self.tokens:
            text += token['form']
            if 'SpaceAfter=No' not in token.get('misc', ''):
                text += " "
        return text.strip()

    def num_tokens(self) -> int:
        return len(self.tokens)

    def has_potential_entities(self) -> bool:
        """Check if sentence likely contains entities"""
        text = self.get_text()

        # Check for organization indicators
        org_indicators = ['Bank', 'AG', 'GmbH', 'SE', 'Landesbank', 'Sparkasse']

        # Check for monetary indicators
        mon_indicators = ['EUR', '€', '%', 'USD']

        # Check for legal indicators
        leg_indicators = ['§', 'Art.', 'Abs.', 'Nr.']

        # Check if any indicator is present
        for indicator in org_indicators + mon_indicators + leg_indicators:
            if indicator in text:
                return True

        # Check for numbers (potential monetary values)
        for token in self.tokens:
            if token['upos'] == 'NUM':
                return True

        return False


def load_sentences(file_path: Path, max_sentences: int = None) -> List[Sentence]:
    """Load all sentences from CoNLL-U file (handles non-standard format where fields are on separate lines)"""
    sentences = []
    current_doc_id = None
    current_sent = None
    field_buffer = []

    print(f"Loading sentences from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:  # Empty line (skip)
                continue

            elif line.startswith('# newdoc'):
                # Save previous sentence if exists
                if current_sent and current_sent.tokens:
                    sentences.append(current_sent)
                    if max_sentences and len(sentences) >= max_sentences:
                        break
                current_doc_id = line.split('=')[-1].strip()
                current_sent = None
                field_buffer = []

            elif line.startswith('#'):
                # Skip other comments
                continue

            else:
                # Check if this looks like a token ID (digit or digit-digit)
                if not field_buffer and line.strip() and (line.strip().isdigit() or '-' in line.strip()):
                    token_id = line.strip()

                    # Skip multi-word tokens
                    if '-' in token_id:
                        continue

                    # Start collecting fields for this token
                    field_buffer = [token_id]

                elif field_buffer:
                    # We're collecting fields for a token
                    field_buffer.append(line)

                    # Once we have all 10 fields, process the token
                    if len(field_buffer) == 10:
                        token_id = field_buffer[0]

                        # Check if this is the start of a new sentence (ID = "1")
                        if token_id == "1":
                            # Save previous sentence if exists
                            if current_sent and current_sent.tokens:
                                sentences.append(current_sent)
                                if max_sentences and len(sentences) >= max_sentences:
                                    break
                            # Start new sentence
                            current_sent = Sentence(current_doc_id)

                        # Add token to current sentence
                        if current_sent is not None:
                            current_sent.add_token(field_buffer)

                        # Reset buffer for next token
                        field_buffer = []

            # Check if we've reached max_sentences
            if max_sentences and len(sentences) >= max_sentences:
                break

    # Don't forget last sentence
    if current_sent and current_sent.tokens:
        sentences.append(current_sent)

    print(f"Loaded {len(sentences)} sentences")
    return sentences


def stratified_sample(sentences: List[Sentence], num_samples: int, seed: int = 42) -> List[Sentence]:
    """Sample sentences with stratification"""
    random.seed(seed)

    print("\nApplying stratified sampling...")

    # Categorize sentences
    short_sentences = []  # < 10 tokens
    medium_sentences = []  # 10-30 tokens
    long_sentences = []  # > 30 tokens
    entity_rich = []  # Likely contains entities

    for sent in sentences:
        num_tokens = sent.num_tokens()

        if sent.has_potential_entities():
            entity_rich.append(sent)

        if num_tokens < 10:
            short_sentences.append(sent)
        elif num_tokens <= 30:
            medium_sentences.append(sent)
        else:
            long_sentences.append(sent)

    print(f"  Short sentences (< 10 tokens): {len(short_sentences)}")
    print(f"  Medium sentences (10-30 tokens): {len(medium_sentences)}")
    print(f"  Long sentences (> 30 tokens): {len(long_sentences)}")
    print(f"  Entity-rich sentences: {len(entity_rich)}")

    # Sample proportionally
    # 70% entity-rich, 30% random
    num_entity_rich = int(num_samples * 0.7)
    num_random = num_samples - num_entity_rich

    sampled = []

    # Sample entity-rich sentences
    if len(entity_rich) >= num_entity_rich:
        sampled.extend(random.sample(entity_rich, num_entity_rich))
    else:
        sampled.extend(entity_rich)
        num_random += (num_entity_rich - len(entity_rich))

    # Sample remaining from all sentences (excluding already sampled)
    remaining = [s for s in sentences if s not in sampled]
    if len(remaining) >= num_random:
        sampled.extend(random.sample(remaining, num_random))
    else:
        sampled.extend(remaining)

    # Shuffle final sample
    random.shuffle(sampled)

    print(f"\nSampled {len(sampled)} sentences")
    return sampled


def save_as_conllu(sentences: List[Sentence], output_path: Path):
    """Save sentences in CoNLL-U format with empty 11th column for NER tags"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(sentences, 1):
            # Write sentence metadata as comments
            f.write(f"# sent_id = {sent_idx}\n")
            f.write(f"# doc_id = {sent.doc_id}\n")
            f.write(f"# text = {sent.get_text()}\n")

            # Write tokens with 11 columns (10 standard CoNLL-U + 1 for NER)
            for token in sent.tokens:
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
                    '_'  # 11th column for NER tag (empty, to be filled)
                ])
                f.write(line + '\n')
            f.write('\n')

    print(f"Saved CoNLL-U format to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample sentences for manual annotation")
    parser.add_argument("--input", required=True, help="Input CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num_sentences", type=int, default=150, help="Number of sentences to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAMPLING SENTENCES FOR MANUAL ANNOTATION")
    print("=" * 60)

    # Load all sentences (or first 50000 for speed)
    sentences = load_sentences(input_path, max_sentences=50000)

    # Sample sentences
    sampled = stratified_sample(sentences, args.num_sentences, args.seed)

    # Save CoNLL-U file
    print("\nSaving sampled sentences...")
    output_file = output_dir / "sample_sentences.conllu"
    save_as_conllu(sampled, output_file)

    # Print preview
    print("\nPreview (first 3 sentences):")
    for i, sent in enumerate(sampled[:3], 1):
        print(f"  {i}. ({sent.num_tokens()} tokens): {sent.get_text()[:80]}...")

    print(f"\nSaved {len(sampled)} sentences to: {output_file}")
    print("The 11th column is empty and ready for NER tags (B-ORG, I-ORG, B-MON, I-MON, B-LEG, I-LEG, O)")
    print("Done!")


if __name__ == "__main__":
    main()
