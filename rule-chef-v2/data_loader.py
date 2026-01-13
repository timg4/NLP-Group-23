"""Minimal CoNLL-U reader for NER spans."""
from typing import Dict, List, Iterator, Tuple, Optional


def load_data(path: str) -> List[Tuple[str, List[Dict]]]:
    """
    Load all sentences with their spans from a CoNLL-U file.

    Returns: [(text, spans), ...] where spans = [{"label": "ORG", "start": X, "end": Y, "text": "..."}, ...]
    """
    return list(_read_sentences(path))


def load_data_by_label(path: str) -> Dict[str, List[Tuple[str, List[Dict]]]]:
    """
    Load data grouped by label for training separate models.

    Returns: {
        "ORG": [(text, [{"start": X, "end": Y, "text": "..."}]), ...],
        "MON": [...],
        "LEG": [...]
    }
    """
    data = {"ORG": [], "MON": [], "LEG": []}

    for text, spans in _read_sentences(path):
        for label in data.keys():
            label_spans = [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in spans if s["label"] == label
            ]
            # Include sentence even if no spans (negative examples are important)
            data[label].append((text, label_spans))

    return data


def _read_sentences(path: str) -> Iterator[Tuple[str, List[Dict]]]:
    """
    Yield (text, spans) tuples from CoNLL-U file.

    Handles document-level character offsets by aligning tokens to sentence text.
    """
    sent_text: Optional[str] = None
    tokens: List[Tuple[str, str]] = []  # (form, bio_tag)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                # End of sentence
                if sent_text and tokens:
                    spans = _tokens_to_spans(tokens, sent_text)
                    yield sent_text, spans
                sent_text = None
                tokens = []
                continue

            if line.startswith("# text ="):
                sent_text = line.split("=", 1)[1].strip()
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")

            # Skip multi-word tokens (e.g., "1-2")
            if "-" in cols[0] or "." in cols[0]:
                continue

            form = cols[1]
            bio = cols[-1].strip()  # BIO tag is in last column
            tokens.append((form, bio))

        # Final sentence
        if sent_text and tokens:
            spans = _tokens_to_spans(tokens, sent_text)
            yield sent_text, spans


def _tokens_to_spans(tokens: List[Tuple[str, str]], text: str) -> List[Dict]:
    """
    Convert token sequence with BIO tags to character spans.

    Aligns tokens to sentence text to compute sentence-relative character offsets.
    """
    # First, align tokens to text to get character positions
    aligned = _align_tokens(tokens, text)

    # Then convert BIO sequences to spans
    spans = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for form, start, end, bio in aligned:
        if bio == "O":
            if cur_label:
                spans.append(_make_span(cur_label, cur_start, cur_end, text))
                cur_label = cur_start = cur_end = None
            continue

        if "-" not in bio:
            # Malformed tag, skip
            continue

        prefix, label = bio.split("-", 1)

        if prefix == "B":
            # Start new entity
            if cur_label:
                spans.append(_make_span(cur_label, cur_start, cur_end, text))
            cur_label, cur_start, cur_end = label, start, end
        elif prefix == "I" and cur_label == label:
            # Continue current entity
            cur_end = end
        else:
            # I-tag without matching B, or different label - start new
            if cur_label:
                spans.append(_make_span(cur_label, cur_start, cur_end, text))
            cur_label, cur_start, cur_end = label, start, end

    # Don't forget last entity
    if cur_label:
        spans.append(_make_span(cur_label, cur_start, cur_end, text))

    return spans


def _align_tokens(tokens: List[Tuple[str, str]], text: str) -> List[Tuple[str, int, int, str]]:
    """
    Align tokens to text, returning (form, start, end, bio) tuples with sentence-relative offsets.
    """
    aligned = []
    pos = 0

    for form, bio in tokens:
        # Skip whitespace
        while pos < len(text) and text[pos].isspace():
            pos += 1

        if pos >= len(text):
            break

        # Try to find token at current position
        if text[pos:].startswith(form):
            start = pos
            end = pos + len(form)
            aligned.append((form, start, end, bio))
            pos = end
        else:
            # Token not found at expected position - try fuzzy match
            # This handles cases where whitespace/newlines were normalized
            idx = text.find(form, pos)
            if idx != -1 and idx < pos + 20:  # Allow some slack
                start = idx
                end = idx + len(form)
                aligned.append((form, start, end, bio))
                pos = end
            else:
                # Skip this token if we can't find it
                pass

    return aligned


def _make_span(label: str, start: int, end: int, text: str) -> Dict:
    """Create span dict with extracted text."""
    return {
        "label": label,
        "start": start,
        "end": end,
        "text": text[start:end]
    }


if __name__ == "__main__":
    # Quick test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/manual_annotation2/my_labels.conllu"

    data = load_data(path)
    print(f"Loaded {len(data)} sentences")

    # Show some examples with entities
    for text, spans in data[:20]:
        if spans:
            print(f"\nText: {text[:80]}...")
            for sp in spans:
                print(f"  {sp['label']}: '{sp['text']}' [{sp['start']}:{sp['end']}]")
