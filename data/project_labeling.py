#!/usr/bin/env python
"""
Project NER Labeling (spaCy + Regex)
-------------------------------------------------------------
Apply the project's spaCy + regex NER labeling method to specific sentences

Usage:  python data/project_labeling.py \
        --input data/manual_annotation/sample_sentences.conllu \
        --output results/labeling_comparison/project_predictions.conllu
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import spacy

# Import the annotation functions from ner_annotation
import sys
sys.path.append(str(Path(__file__).parent))

# Entity patterns (same as ner_annotation.py)
MONETARY_PATTERNS = [
    r'\d{1,3}(?:[.,]\d{3})*(?:,\d{2})?\s*(?:EUR|USD|€|\$|CHF)',
    r'(?:EUR|USD|€|\$|CHF)\s*\d{1,3}(?:[.,]\d{3})*(?:,\d{2})?',
    r'\d+(?:,\d+)?\s*%',
]

LEGAL_PATTERNS = [
    r'§\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?',
    r'Art\.\s*\d+[a-z]?(?:\s+[A-ZÄÖÜß]+)?',
    r'Abs\.\s*\d+[a-z]?',
    r'Nr\.\s*\d+',
]

MONETARY_RX = re.compile('|'.join(f'({p})' for p in MONETARY_PATTERNS), re.IGNORECASE)
LEGAL_RX = re.compile('|'.join(f'({p})' for p in LEGAL_PATTERNS))


def load_conllu_sentences(file_path: Path) -> List[Dict]:
    """Load sentences from CoNLL-U file"""
    sentences = []
    current_sent = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:  # Sentence boundary
                if current_sent and current_sent['tokens']:
                    sentences.append(current_sent)
                current_sent = None

            elif line.startswith('# sent_id'):
                current_sent = {
                    'sent_id': line.split('=')[-1].strip(),
                    'doc_id': None,
                    'text': None,
                    'tokens': []
                }

            elif line.startswith('# doc_id') and current_sent:
                current_sent['doc_id'] = line.split('=')[-1].strip()

            elif line.startswith('# text') and current_sent:
                current_sent['text'] = line.split('=', 1)[-1].strip()

            elif not line.startswith('#') and current_sent:
                fields = line.split('\t')
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
                    current_sent['tokens'].append(token)

    if current_sent and current_sent['tokens']:
        sentences.append(current_sent)

    return sentences


def build_char_to_token_mapping(tokens: List[Dict]) -> Tuple[Dict[int, int], str]:
    """Build mapping from character position to token index"""
    mapping = {}
    text = ""
    char_pos = 0

    for token_idx, token in enumerate(tokens):
        form = token['form']
        token_len = len(form)

        for i in range(token_len):
            mapping[char_pos + i] = token_idx

        text += form
        char_pos += token_len

        if 'SpaceAfter=No' not in token.get('misc', ''):
            text += " "
            char_pos += 1

    return mapping, text.strip()


def span_to_tokens(start: int, end: int, char_to_token: Dict[int, int]) -> Tuple[int, int]:
    """Convert character span to token indices"""
    token_start = None
    token_end = None

    for char_idx in range(start, end):
        if char_idx in char_to_token:
            token_idx = char_to_token[char_idx]
            if token_start is None:
                token_start = token_idx
            token_end = token_idx

    if token_start is not None and token_end is not None:
        return token_start, token_end
    return None, None


def annotate_with_spacy(tokens: List[Dict], nlp, char_to_token: Dict[int, int], text: str) -> List[Tuple[int, int, str]]:
    """Use spaCy to find ORG entities"""
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ == 'ORG':
            token_start, token_end = span_to_tokens(ent.start_char, ent.end_char, char_to_token)
            if token_start is not None:
                entities.append((token_start, token_end, 'ORG'))

    return entities


def annotate_with_regex(pattern, entity_type: str, char_to_token: Dict[int, int], text: str) -> List[Tuple[int, int, str]]:
    """Use regex to find entities"""
    entities = []

    for match in pattern.finditer(text):
        token_start, token_end = span_to_tokens(match.start(), match.end(), char_to_token)
        if token_start is not None:
            entities.append((token_start, token_end, entity_type))

    return entities


def entities_to_bio(num_tokens: int, entities: List[Tuple[int, int, str]]) -> List[str]:
    """Convert entity spans to BIO tags"""
    tags = ['O'] * num_tokens

    sorted_entities = sorted(entities, key=lambda x: (x[0], -x[1]))

    for start, end, ent_type in sorted_entities:
        if tags[start] != 'O':
            continue

        tags[start] = f'B-{ent_type}'
        for i in range(start + 1, end + 1):
            if i < num_tokens and tags[i] == 'O':
                tags[i] = f'I-{ent_type}'

    return tags


def annotate_sentence(sentence: Dict, nlp) -> List[str]:
    """Annotate a sentence with NER tags"""
    tokens = sentence['tokens']

    if not tokens:
        return []

    char_to_token, text = build_char_to_token_mapping(tokens)

    entities = []

    # Get ORG entities from spaCy
    entities.extend(annotate_with_spacy(tokens, nlp, char_to_token, text))

    # Get MON entities from regex
    entities.extend(annotate_with_regex(MONETARY_RX, 'MON', char_to_token, text))

    # Get LEG entities from regex
    entities.extend(annotate_with_regex(LEGAL_RX, 'LEG', char_to_token, text))

    # Convert to BIO tags
    bio_tags = entities_to_bio(len(tokens), entities)

    return bio_tags


def save_predictions(sentences: List[Dict], predictions: List[List[str]], output_path: Path):
    """Save predictions in CoNLL-U format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent, tags in zip(sentences, predictions):
            # Write comments
            f.write(f"# sent_id = {sent['sent_id']}\n")
            if sent['doc_id']:
                f.write(f"# doc_id = {sent['doc_id']}\n")
            if sent['text']:
                f.write(f"# text = {sent['text']}\n")

            # Write tokens with NER tags
            for token, tag in zip(sent['tokens'], tags):
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
                    tag  # 11th column: NER tag
                ])
                f.write(line + '\n')

            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Apply project NER labeling (spaCy + regex)")
    parser.add_argument("--input", required=True, help="Input CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output CoNLL-U file with NER tags")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("Project NER Labeling (spaCy + Regex)")
    print("=" * 60)

    # Load spaCy model
    print("\nLoading spaCy German model...")
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        print("ERROR: spaCy model not found!")
        print("Download with: python -m spacy download de_core_news_lg")
        exit(1)

    # Load sentences
    print(f"\nLoading sentences from: {input_path}")
    sentences = load_conllu_sentences(input_path)
    print(f"Loaded {len(sentences)} sentences")

    # Annotate sentences
    print("\nAnnotating sentences...")
    predictions = []

    for sent in tqdm(sentences, desc="Sentences"):
        tags = annotate_sentence(sent, nlp)
        predictions.append(tags)

    # Save predictions
    save_predictions(sentences, predictions, output_path)
    print(f"\nSaved predictions to: {output_path}")

    # Statistics
    entity_counts = {'ORG': 0, 'MON': 0, 'LEG': 0}
    for tags in predictions:
        for tag in tags:
            if tag.startswith('B-'):
                ent_type = tag[2:]
                entity_counts[ent_type] = entity_counts.get(ent_type, 0) + 1

    print("\nEntity counts:")
    for ent_type, count in entity_counts.items():
        print(f"  {ent_type}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
