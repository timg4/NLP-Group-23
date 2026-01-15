#!/usr/bin/env python
"""
CRF-Based NER Baseline
-------------------------------------------------------------
Conditional Random Fields implementation with comprehensive feature engineering
for German financial/legal NER

Features include:
- Word-level features (shape, capitalization, etc.)
- Morphological features from CoNLL-U (POS, lemma, etc.)
- Context features (±2 window)
- Domain-specific features (currency symbols, legal terms, etc.)
"""

import re
from typing import List, Dict, Tuple
import sklearn_crfsuite
from sklearn_crfsuite import metrics


class CRFNER:
    """CRF-based NER system with feature engineering"""

    def __init__(self, c1=0.1, c2=0.1, max_iterations=100):
        """
        Initialize CRF model

        Args:
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: Maximum training iterations
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )

    def _word_shape(self, word: str) -> str:
        """
        Get word shape pattern
        'Deutsche' -> 'Xxxxxxxx' (X=upper, x=lower, d=digit)
        """
        if len(word) > 20:
            return 'long'

        shape = ''
        for char in word:
            if char.isupper():
                shape += 'X'
            elif char.islower():
                shape += 'x'
            elif char.isdigit():
                shape += 'd'
            else:
                shape += char
        return shape

    def _parse_feats(self, feats_str: str) -> Dict[str, str]:
        """Parse morphological features from CoNLL-U format"""
        if feats_str == '_':
            return {}

        feat_dict = {}
        for feat in feats_str.split('|'):
            if '=' in feat:
                key, value = feat.split('=', 1)
                feat_dict[key] = value
        return feat_dict

    def _token_features(self, sentence: List[Dict], i: int) -> Dict[str, any]:
        """
        Extract features for token at position i in sentence

        Args:
            sentence: List of token dictionaries
            i: Token index

        Returns:
            Dictionary of features
        """
        token = sentence[i]
        word = token['form']
        lemma = token['lemma']
        upos = token['upos']
        xpos = token['xpos']
        feats_str = token['feats']

        features = {
            'bias': 1.0,
            'word.lower': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:3]': word[:3],
            'word[:2]': word[:2],
            'word.isupper': word.isupper(),
            'word.istitle': word.istitle(),
            'word.isdigit': word.isdigit(),
            'word.shape': self._word_shape(word),
            'word.length': min(len(word), 20),  # Cap at 20 to avoid too many features
            'lemma': lemma,
            'upos': upos,
            'xpos': xpos,
            'upos[:2]': upos[:2] if len(upos) >= 2 else upos,
        }

        # Parse morphological features
        morph_feats = self._parse_feats(feats_str)
        for key, value in morph_feats.items():
            features[f'morph.{key}'] = value

        # Domain-specific features
        features['has_currency'] = any(c in word for c in ['€', '$', 'EUR', 'USD', 'CHF'])
        features['has_paragraph'] = '§' in word
        features['has_percent'] = '%' in word
        features['has_period'] = '.' in word
        features['has_comma'] = ',' in word
        features['has_hyphen'] = '-' in word

        # Company suffixes
        features['ends_with_AG'] = word.endswith('AG')
        features['ends_with_GmbH'] = word.endswith('GmbH')
        features['ends_with_SE'] = word.endswith('SE')

        # Legal abbreviations
        legal_abbrevs = ['Art.', 'Abs.', 'Nr.', 'Kap.', 'Anm.']
        features['is_legal_abbrev'] = word in legal_abbrevs

        # Bank/finance keywords
        bank_keywords = ['Bank', 'Landesbank', 'Sparkasse', 'Bundesbank']
        features['is_bank_keyword'] = word in bank_keywords

        # Number format patterns
        features['german_number_format'] = bool(re.match(r'\d{1,3}(?:\.\d{3})+(?:,\d{2})?', word))
        features['has_multiple_dots'] = word.count('.') > 1
        features['digit_comma_digit'] = bool(re.search(r'\d,\d', word))

        # Position features
        features['position'] = i / len(sentence) if len(sentence) > 0 else 0
        features['is_first'] = (i == 0)
        features['is_last'] = (i == len(sentence) - 1)

        # Context features: Previous token
        if i > 0:
            prev_token = sentence[i - 1]
            features['prev_word.lower'] = prev_token['form'].lower()
            features['prev_upos'] = prev_token['upos']
            features['prev_xpos'] = prev_token['xpos']
            features['prev_word.shape'] = self._word_shape(prev_token['form'])
            features['prev_word.istitle'] = prev_token['form'].istitle()
        else:
            features['BOS'] = True  # Beginning of sentence

        # Context features: Next token
        if i < len(sentence) - 1:
            next_token = sentence[i + 1]
            features['next_word.lower'] = next_token['form'].lower()
            features['next_upos'] = next_token['upos']
            features['next_xpos'] = next_token['xpos']
            features['next_word.shape'] = self._word_shape(next_token['form'])
            features['next_word.istitle'] = next_token['form'].istitle()
        else:
            features['EOS'] = True  # End of sentence

        # Context features: ±2 window
        if i > 1:
            features['prev2_upos'] = sentence[i - 2]['upos']

        if i < len(sentence) - 2:
            features['next2_upos'] = sentence[i + 2]['upos']

        # Bigram features
        if i > 0:
            features['prev_word+word'] = f"{sentence[i-1]['form'].lower()}+{word.lower()}"

        if i < len(sentence) - 1:
            features['word+next_word'] = f"{word.lower()}+{sentence[i+1]['form'].lower()}"

        return features

    def sentence_features(self, sentence: List[Dict]) -> List[Dict]:
        """Extract features for all tokens in a sentence"""
        return [self._token_features(sentence, i) for i in range(len(sentence))]

    def fit(self, X_train: List[List[Dict]], y_train: List[List[str]]):
        """
        Train CRF model

        Args:
            X_train: List of sentences (each sentence is a list of feature dicts)
            y_train: List of tag sequences
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: List[List[Dict]]) -> List[List[str]]:
        """
        Predict tags for test sentences

        Args:
            X_test: List of sentences (each sentence is a list of feature dicts)

        Returns:
            List of predicted tag sequences
        """
        return self.model.predict(X_test)

    def get_feature_weights(self, top_n=20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features for each entity type

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary mapping entity types to (feature, weight) tuples
        """
        try:
            from collections import Counter

            # Get state (tag) features
            weights = {}
            state_features = Counter(self.model.state_features_).most_common()

            # Group by entity type
            for (tag, feat), weight in state_features[:100]:
                if tag not in weights:
                    weights[tag] = []
                weights[tag].append((feat, weight))

            # Keep top N for each tag
            for tag in weights:
                weights[tag] = weights[tag][:top_n]

            return weights
        except:
            return {}


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
        print("Usage: python crf_ner.py <train.conllu>")
        sys.exit(1)

    train_file = sys.argv[1]
    print(f"Loading {train_file}...")

    sentences = load_conllu(train_file)
    print(f"Loaded {len(sentences)} sentences")

    # Prepare data
    print("\nExtracting features...")
    crf = CRFNER()

    X_train = [crf.sentence_features(tokens) for tokens, _ in sentences[:100]]
    y_train = [tags for _, tags in sentences[:100]]

    print(f"Training on {len(X_train)} sentences...")
    crf.fit(X_train, y_train)

    print("\nDone! Model trained.")

    # Show example predictions
    X_test = [crf.sentence_features(tokens) for tokens, _ in sentences[:5]]
    y_pred = crf.predict(X_test)

    print("\nExample predictions:")
    for i, (tokens, gold_tags) in enumerate(sentences[:5]):
        print(f"\n--- Sentence {i + 1} ---")
        print(f"{'TOKEN':20} {'GOLD':10} {'PRED':10}")
        print("-" * 40)
        for token, gold, pred in zip(tokens, gold_tags, y_pred[i]):
            print(f"{token['form']:20} {gold:10} {pred:10}")
