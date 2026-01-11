#this converts gold BIO annotations to character spans

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from src.data.conllu_reader import Sentence, Token

@dataclass(frozen=True)
class CharToken:
    form: str
    start: int
    end: int
    bio: str

def align_tokens_to_text(sent: Sentence) -> List[CharToken]:
    text = sent.text
    i = 0
    out: List[CharToken] = []

    for tok in sent.tokens:
        while i < len(text) and text[i].isspace():
            i += 1

        form = tok.form
        if not text.startswith(form, i):
            # fail loudly: you WANT to know if alignment breaks
            snippet = text[max(0, i-20):min(len(text), i+40)]
            raise ValueError(
                f"Token-text mismatch in sent_id={sent.sent_id}. "
                f"At pos {i}, expected '{form}', context='{snippet}'"
            )

        start = i
        end = i + len(form)
        out.append(CharToken(form=form, start=start, end=end, bio=tok.bio))
        i = end

    return out

def bio_to_spans(char_tokens: List[CharToken]) -> List[Dict]:
    spans = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def close_span():
        nonlocal cur_label, cur_start, cur_end
        if cur_label is not None:
            spans.append({"label": cur_label, "start": cur_start, "end": cur_end})
        cur_label = None
        cur_start = None
        cur_end = None

    for t in char_tokens:
        tag = t.bio
        if tag == "O":
            close_span()
            continue

        if "-" not in tag:
            raise ValueError(f"Bad BIO tag '{tag}'")

        pref, lab = tag.split("-", 1)

        if pref == "B":
            close_span()
            cur_label = lab
            cur_start = t.start
            cur_end = t.end
        elif pref == "I":
            
            if cur_label is None:

                cur_label = lab
                cur_start = t.start
                cur_end = t.end
            else:
                if lab != cur_label:
                    close_span()
                    cur_label = lab
                    cur_start = t.start
                    cur_end = t.end
                else:
                    cur_end = t.end
        else:
            raise ValueError(f"Unknown BIO prefix '{pref}' in '{tag}'")

    close_span()
    return spans