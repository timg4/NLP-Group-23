#!/usr/bin/env python
"""
Project NER Labeling (spaCy + Regex)
-------------------------------------------------------------
Apply the project's spaCy + regex NER labeling method to specific sentences

Usage:  python data/project_labeling.py --input data/manual_annotation/sample_sentences.conllu --output results/labeling_comparison/project_predictions.conllu
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

# ============================================================================
# STRATEGY 2: Organization Keywords
# ============================================================================
ORG_KEYWORDS = {
    'Bank', 'BANK', 'Banking', 'Sparkasse', 'Sparkassen',
    'Landesbank', 'Bundesbank', 'Genossenschaftsbank', 'Girozentrale',
    'AG', 'GmbH', 'GbR', 'Ltd.', 'Ltd', 'SE', 'KGaA',
    'eG', 'e.V.', 'OGAW', 'OGA', 'Union',
    'Investors', 'Services', 'Service', 'Rating', 'Ratings',
    'Credit', 'Clearing', 'Europe',
    'DZ', 'UniCredit', 'Helaba', 'S&P', 'Standard', 'Moody', 'Clearstream'
}

# ============================================================================
# STRATEGY 3: Enhanced Monetary Detection
# ============================================================================
CURRENCY_TOKENS = {
    'EUR', 'Euro', '€', 'USD', 'CHF', 'GBP', '$', 'US-Dollar', 'US$'
}

MAGNITUDE_WORDS = {
    'Million', 'Millionen', 'Mio.', 'Mio',
    'Milliarde', 'Milliarden', 'Mrd.', 'Mrd',
    'tausend', 'Tausend', 'Tsd.', 'Tsd'
}

# ============================================================================
# STRATEGY 1: Legal Reference Detection (Token-Level)
# ============================================================================
LEGAL_START_TOKENS = {'§', '§§', 'Art.', 'Art', 'Artikel', 'Nr.', 'Nr', 'Abs.', 'Abs', 'Absatz'}


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


def annotate_orgs_with_keywords(tokens: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    STRATEGY 2: Keyword-based organization detection
    Find PROPN spans that contain organization keywords
    """
    entities = []
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]
        # Check if this could start an organization
        if token['upos'] == 'PROPN' or token['upos'] in {'X', 'SYM'}:
            # Collect consecutive PROPN/proper tokens
            span_start = i
            span_tokens = []
            j = i

            while j < n:
                t = tokens[j]
                # Include PROPN, proper nouns, symbols, and connecting punctuation
                if t['upos'] in {'PROPN', 'X', 'SYM'} or \
                   (t['upos'] == 'NOUN' and t['form'] in ORG_KEYWORDS) or \
                   (t['form'] in {'&', '-', '–', '.'}):
                    span_tokens.append(t)
                    j += 1
                else:
                    break

            # Check if span contains any organization keyword
            has_keyword = any(t['form'] in ORG_KEYWORDS or t['lemma'] in ORG_KEYWORDS
                            for t in span_tokens)

            if has_keyword and span_tokens:
                span_end = span_start + len(span_tokens) - 1
                entities.append((span_start, span_end, 'ORG'))

            i = j
        else:
            i += 1

    return entities


def is_numeric_token(token: Dict) -> bool:
    """Check if token represents a number"""
    if token['upos'] == 'NUM':
        return True
    # Check if form is numeric
    form = token['form'].replace('.', '').replace(',', '')
    return form.isdigit() and len(form) > 0


def is_currency_token(form: str) -> bool:
    """Check if token is a currency symbol or word"""
    return form in CURRENCY_TOKENS or \
           form.startswith('EUR') or form.endswith('EUR') or \
           form in {'€', '$'}


def annotate_monetary_enhanced(tokens: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    STRATEGY 3: Enhanced monetary detection with magnitude words
    """
    entities = []
    n = len(tokens)

    # Find currency tokens and link with numbers
    for i, token in enumerate(tokens):
        if is_currency_token(token['form']):
            start = i
            end = i

            # Look backward for number
            if i > 0 and is_numeric_token(tokens[i - 1]):
                start = i - 1
            # Look forward for number
            elif i + 1 < n and is_numeric_token(tokens[i + 1]):
                end = i + 1

            # Look for magnitude words after the amount
            if end + 1 < n and (tokens[end + 1]['form'] in MAGNITUDE_WORDS or
                                tokens[end + 1]['lemma'] in MAGNITUDE_WORDS):
                end = end + 1

            entities.append((start, end, 'MON'))

    # Find percentages
    for i, token in enumerate(tokens):
        if '%' in token['form'] or token['lemma'].lower() == 'prozent':
            # Look backward for number
            if i > 0 and is_numeric_token(tokens[i - 1]):
                entities.append((i - 1, i, 'MON'))
            elif '%' in token['form'] and is_numeric_token(token):
                entities.append((i, i, 'MON'))

    return entities


def is_legal_start(form: str, lemma: str) -> bool:
    """Check if token can start a legal reference"""
    return form in LEGAL_START_TOKENS or \
           lemma in {'Artikel', 'Nummer', 'Paragraph', 'Absatz'} or \
           form.startswith('§')


def is_legal_continuation(form: str, lemma: str, upos: str) -> bool:
    """Check if token can continue a legal reference"""
    # Numbers are very common in legal references
    if upos == 'NUM':
        return True

    # Punctuation used in legal refs
    if re.fullmatch(r'[\(\)\[\],\-–;/]', form):
        return True

    # Alphanumeric patterns like "15a" or "12b"
    if re.fullmatch(r'\d+[a-zA-Z]*', form):
        return True

    # Legal reference words
    if form in {'Abs.', 'Abs', 'Absatz', 'Nr.', 'Nr', 'Satz', 'S.', 'lit.', 'lit', 'Art.', 'Art'}:
        return True

    # ALL CAPS abbreviations (like "PVO", "EG")
    if len(form) >= 2 and form.isupper() and form.isalpha():
        return True

    # Connecting words
    if form in {'und', 'oder', 'bis'}:
        return True

    # Legal reference keywords in lemma
    if lemma in {'Verordnung', 'Richtlinie', 'Gesetz'}:
        return True

    return False


def annotate_legal_token_level(tokens: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    STRATEGY 1: Token-level legal reference detection
    Detects start of legal reference and continues while conditions are met
    """
    entities = []
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]

        # Check if this starts a legal reference
        if is_legal_start(token['form'], token['lemma']):
            start = i
            j = i + 1

            # Continue while we see legal reference continuation tokens
            while j < n and is_legal_continuation(tokens[j]['form'],
                                                   tokens[j]['lemma'],
                                                   tokens[j]['upos']):
                j += 1

            end = j - 1
            entities.append((start, end, 'LEG'))
            i = j
        else:
            i += 1

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
    """Annotate a sentence with NER tags using improved strategies"""
    tokens = sentence['tokens']

    if not tokens:
        return []

    char_to_token, text = build_char_to_token_mapping(tokens)

    entities = []

    # STRATEGY 2: Get ORG entities from spaCy (existing method)
    entities.extend(annotate_with_spacy(tokens, nlp, char_to_token, text))

    # STRATEGY 2: Get ORG entities from keyword-based detection (NEW)
    entities.extend(annotate_orgs_with_keywords(tokens))

    # STRATEGY 3: Get MON entities with enhanced detection (NEW)
    entities.extend(annotate_monetary_enhanced(tokens))

    # STRATEGY 1: Get LEG entities with token-level detection (NEW)
    entities.extend(annotate_legal_token_level(tokens))

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
