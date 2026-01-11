#this reads the CoNLL-U formatted files with BIO annotations to match the format we need

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator

@dataclass
class Token:
    id: int
    form: str
    misc: Dict[str, str]
    bio: str

@dataclass
class Sentence:
    sent_id: str
    doc_id: Optional[str]
    text: str
    tokens: List[Token]

def _parse_misc(misc_str: str) -> Dict[str, str]:
    if misc_str == "_" or misc_str.strip() == "":
        return {}
    parts = misc_str.split("|")
    misc = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            misc[k] = v
        else:
            misc[p] = "true"
    return misc

def read_conllu(path: str) -> Iterator[Sentence]:
    sent_id = None
    doc_id = None
    text = None
    tokens: List[Token] = []

    def flush():
        nonlocal sent_id, doc_id, text, tokens
        if text is not None and tokens:
            yield Sentence(
                sent_id=str(sent_id) if sent_id is not None else "",
                doc_id=doc_id,
                text=text,
                tokens=tokens,
            )
        sent_id = None
        doc_id = None
        text = None
        tokens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                yield from flush()
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id ="):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# doc_id ="):
                    doc_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    text = line.split("=", 1)[1].strip()
                continue

            cols = line.split("\t")
            
            if "-" in cols[0] or "." in cols[0]:
                continue

            tid = int(cols[0])
            form = cols[1]
            misc = _parse_misc(cols[9]) if len(cols) > 9 else {}
            bio = cols[-1].strip()
            tokens.append(Token(id=tid, form=form, misc=misc, bio=bio))

 
        yield from flush()