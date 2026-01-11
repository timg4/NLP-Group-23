from typing import Dict
from src.data.conllu_reader import read_conllu
from src.data.gold_to_spans import align_tokens_to_text, bio_to_spans

LABELS = {"ORG", "MON", "LEG"}

def _pick(obj, *names):
    #dict-like
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        raise KeyError(f"None of keys {names} found in dict: {obj}")

    #object-like
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)

    # dataclass-like
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
        for n in names:
            if n in d:
                return d[n]

    raise KeyError(f"None of attrs {names} found in object of type {type(obj)}")

def load_gold(path: str) -> Dict[str, Dict]:
    data = {}
    for sent in read_conllu(path):
        key = f"{sent.doc_id}:{sent.sent_id}" if sent.doc_id is not None else str(sent.sent_id)

        char_tokens = align_tokens_to_text(sent)
        spans = bio_to_spans(char_tokens)

        tokens = []
        offsets = []
        for ct in char_tokens:
            tok = _pick(ct, "form", "token", "text")
            start = _pick(ct, "start", "start_char")
            end = _pick(ct, "end", "end_char")
            tokens.append(tok)
            offsets.append((start, end))

        for sp in spans:
            sp["text"] = sent.text[sp["start"]:sp["end"]]

        data[key] = {"text": sent.text, "gold_spans": spans, "tokens": tokens, "offsets": offsets}
    return data