#!/usr/bin/env python
"""
Rule-Based NER Baseline
-------------------------------------------------------------
Pattern-matching system using regular expressions and dictionaries
for German financial/legal NER

Entity types:
- ORGANIZATION (ORG): Companies, banks, institutions
- MONETARY (MON): Currency amounts, percentages
- LEGAL_REFERENCE (LEG): Paragraph and article references
"""

import re
from typing import List, Dict, Tuple


class RuleBasedNER:
    """Rule-based NER system using regex patterns"""

    def __init__(self):
        # Compile patterns for each entity type
        self.org_patterns = self._compile_org_patterns()
        self.mon_patterns = self._compile_mon_patterns()
        self.leg_patterns = self._compile_leg_patterns()

    def _compile_org_patterns(self):
        """Compile regex patterns for ORGANIZATION entities"""
        patterns = [
            # Company legal forms: "Deutsche Bank AG", "LBBW GmbH"
            r'(?:[A-ZÄÖÜ][a-zäöüß]+\s+){1,5}(?:AG|GmbH|SE|KGaA|e\.V\.|mbH)',
            # Bank keywords: "Landesbank Baden-Württemberg"
            r'(?:[A-ZÄÖÜ][a-zäöüß]+\s+){0,3}(?:Landesbank|Sparkasse|Bank|Bundesbank)(?:\s+[A-ZÄÖÜ][a-zäöüß-]+){0,3}',
            # All-caps sequences: "DEUTSCHE BANK", "LBBW"
            r'[A-ZÄÖÜ]{2,}(?:\s+[A-ZÄÖÜ]{2,}){0,3}(?:\s+(?:AG|SE|GmbH))?',
        ]
        return [re.compile(p) for p in patterns]

    def _compile_mon_patterns(self):
        """Compile regex patterns for MONETARY entities"""
        patterns = [
            # Number + currency: "24.000.000 EUR", "100,00 €"
            r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,4})?\s*(?:EUR|USD|€|\$|CHF)',
            # Currency + number: "EUR 500.000"
            r'(?:EUR|USD|€|\$|CHF)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,4})?',
            # Percentages: "3,41 %"
            r'\d+(?:[.,]\d+)?\s*%',
            # Standalone monetary amounts in German format
            r'\d{1,3}(?:\.\d{3})+(?:,\d{2})?',
        ]
        return [re.compile(p) for p in patterns]

    def _compile_leg_patterns(self):
        """Compile regex patterns for LEGAL_REFERENCE entities"""
        patterns = [
            # Paragraph sign: "§ 15", "§ 15 Abs. 2"
            r'§\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+[A-Z]+)?',
            # Article: "Art. 12 PVO", "Art. 5"
            r'Art\.\s*\d+[a-z]?(?:\s+[A-ZÄÖÜß]+)?',
            # Absatz: "Abs. 1"
            r'Abs\.\s*\d+[a-z]?',
            # Number: "Nr. 11", "Nr. 397"
            r'Nr\.\s*\d+',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _build_char_to_token_mapping(self, tokens: List[Dict]) -> Tuple[Dict[int, int], str]:
        """Build mapping from character position to token index and reconstruct text"""
        mapping = {}
        text = ""
        char_pos = 0

        for token_idx, token in enumerate(tokens):
            form = token['form']
            token_len = len(form)

            # Map each character to its token
            for i in range(token_len):
                mapping[char_pos + i] = token_idx

            text += form
            char_pos += token_len

            # Add space if not SpaceAfter=No
            if 'SpaceAfter=No' not in token.get('misc', ''):
                text += " "
                char_pos += 1

        return mapping, text.strip()

    def _span_to_tokens(self, start: int, end: int, char_to_token: Dict[int, int]) -> Tuple[int, int]:
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

    def _validate_org(self, tokens: List[Dict], start: int, end: int) -> bool:
        """Validate ORG entity using POS tags"""
        # Must contain at least one NOUN or PROPN
        for i in range(start, end + 1):
            if i < len(tokens):
                upos = tokens[i]['upos']
                if upos in ['NOUN', 'PROPN']:
                    return True
        return False

    def _validate_mon(self, tokens: List[Dict], start: int, end: int) -> bool:
        """Validate MON entity using POS tags"""
        # Must contain at least one NUM token
        for i in range(start, end + 1):
            if i < len(tokens):
                upos = tokens[i]['upos']
                if upos == 'NUM':
                    return True
        return False

    def _find_entities(self, tokens: List[Dict], patterns: List, entity_type: str,
                      char_to_token: Dict[int, int], text: str, validate_func=None) -> List[Tuple[int, int, str]]:
        """Find entities using patterns"""
        entities = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                token_start, token_end = self._span_to_tokens(match.start(), match.end(), char_to_token)

                if token_start is not None:
                    # Apply validation if provided
                    if validate_func is None or validate_func(tokens, token_start, token_end):
                        entities.append((token_start, token_end, entity_type))

        return entities

    def _entities_to_bio(self, num_tokens: int, entities: List[Tuple[int, int, str]]) -> List[str]:
        """Convert entity spans to BIO tags"""
        tags = ['O'] * num_tokens

        # Sort entities by start position, then by length (longest first) to handle overlaps
        sorted_entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))

        for start, end, ent_type in sorted_entities:
            # Skip if already tagged (keep first/longest match)
            if tags[start] != 'O':
                continue

            # Assign BIO tags
            tags[start] = f'B-{ent_type}'
            for i in range(start + 1, end + 1):
                if i < num_tokens and tags[i] == 'O':
                    tags[i] = f'I-{ent_type}'

        return tags

    def predict(self, tokens: List[Dict]) -> List[str]:
        """Predict NER tags for a sentence"""
        if not tokens:
            return []

        # Build character-to-token mapping and reconstruct text
        char_to_token, text = self._build_char_to_token_mapping(tokens)

        # Find entities of each type
        entities = []

        # ORG entities with validation
        entities.extend(self._find_entities(
            tokens, self.org_patterns, 'ORG',
            char_to_token, text, self._validate_org
        ))

        # MON entities with validation
        entities.extend(self._find_entities(
            tokens, self.mon_patterns, 'MON',
            char_to_token, text, self._validate_mon
        ))

        # LEG entities (no validation needed - patterns are specific)
        entities.extend(self._find_entities(
            tokens, self.leg_patterns, 'LEG',
            char_to_token, text
        ))

        # Convert to BIO tags
        bio_tags = self._entities_to_bio(len(tokens), entities)

        return bio_tags


def load_conllu(file_path: str) -> List[Tuple[List[Dict], List[str]]]:
    """Load CoNLL-U file with NER annotations"""
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if not line:  # Empty line = sentence boundary
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                current_tokens = []
                current_tags = []

            elif not line.startswith('#'):
                fields = line.split('\t')
                if len(fields) >= 11 and '-' not in fields[0]:
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
                    current_tokens.append(token)
                    current_tags.append(fields[10].strip())

    # Don't forget last sentence
    if current_tokens:
        sentences.append((current_tokens, current_tags))

    return sentences


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_rulebased_ner.py <input.conllu>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"Loading {input_file}...")

    sentences = load_conllu(input_file)
    print(f"Loaded {len(sentences)} sentences")

    print("\nRunning rule-based NER...")
    ner = RuleBasedNER()

    # Predict on first few sentences as demo
    for i, (tokens, gold_tags) in enumerate(sentences[:5]):
        pred_tags = ner.predict(tokens)

        print(f"\n--- Sentence {i + 1} ---")
        print(f"{'TOKEN':20} {'GOLD':10} {'PRED':10}")
        print("-" * 40)
        for token, gold, pred in zip(tokens, gold_tags, pred_tags):
            print(f"{token['form']:20} {gold:10} {pred:10}")
