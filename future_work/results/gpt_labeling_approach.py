
import re
"""
Python class used by GPT to label data (as labelling with gpt models was not working)
- only here for reference was never run
"""
# Read the original CONLLU file
input_path = "/mnt/data/sample_sentences.conllu"
output_path = "/mnt/data/sample_sentences_labeled.tsv"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

class Tok:
    def __init__(self, sent_id, idx, cols):
        self.sent_id = sent_id
        self.idx = idx
        self.id = int(cols[0])
        self.form = cols[1]
        self.lemma = cols[2]
        self.upos = cols[3]
        self.xpos = cols[4]
        self.feats = cols[5]
        self.head = cols[6]
        self.deprel = cols[7]
        self.misc1 = cols[8]
        self.misc2 = cols[9]

def parse_sentences(lines):
    sents = []
    cur = []
    cur_id = None
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append((cur_id, cur))
                cur = []
                cur_id = None
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                cur_id = line.split("=", 1)[1].strip()
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        tok = Tok(cur_id, len(cur), cols)
        cur.append(tok)
    if cur:
        sents.append((cur_id, cur))
    return sents

sents = parse_sentences(lines)

def legal_start(form, lemma, upos):
    return (
        form in {"§", "§§", "Art.", "Art", "Artikel", "Nr.", "Nr", "Abs.", "Abs", "Absatz"}
        or lemma in {"Artikel", "Nummer", "Paragraph", "Absatz"}
        or form.startswith("§")
    )

def is_all_caps(s):
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)

def legal_continuation(form, lemma, upos):
    if upos == "NUM":
        return True
    if re.fullmatch(r"[\(\)\[\],\-–;/]", form):
        return True
    if re.fullmatch(r"\d+[a-zA-Z]*", form):
        return True
    if form in {"Abs.", "Abs", "Absatz", "Nr.", "Nr", "Satz", "S.", "lit.", "lit"}:
        return True
    if is_all_caps(form) and len(form) >= 2:
        return True
    if re.fullmatch(r"\d+/\d+(/\d+)?", form):
        return True
    if re.fullmatch(r"\d+[a-z]", form):
        return True
    if form in {"und", "oder"}:
        return True
    if lemma in {"Verordnung", "Richtlinie"}:
        return True
    return False

currency_tokens = {
    "EUR", "Euro", "€", "USD", "CHF", "GBP",
    "EUR)", "EUR,", "EUR.", "US-Dollar", "US$"
}

def is_currency(form):
    base = form.strip()
    if base in currency_tokens:
        return True
    if base.endswith("EUR") or base.startswith("EUR"):
        return True
    if base in {"€", "$"}:
        return True
    return False

def is_numeric_token(tok):
    f = tok.form.replace(".", "").replace(",", "")
    if tok.upos == "NUM":
        return True
    if re.fullmatch(r"\d+([.,]\d+)?", tok.form):
        return True
    if f.isdigit() and len(f) > 0:
        return True
    return False

def percent_like(form, lemma):
    if "%" in form:
        return True
    if lemma.lower() in {"prozent", "proz"}:
        return True
    return False

ORG_KEYWORDS = {
    "Bank", "BANK", "Banking", "Sparkasse", "Sparkassen",
    "Landesbank", "Bundesbank", "Genossenschaftsbank", "Girozentrale",
    "AG", "GmbH", "GbR", "Ltd.", "Ltd", "SE", "KGaA",
    "eG", "e.V.", "OGAW", "OGA", "Union",
    "Investors", "Services", "Service", "Rating", "Ratings",
    "Credit", "Clearing", "Europe",
    "DZ", "UniCredit", "Helaba", "S&P", "Standard", "Moody", "Clearstream"
}

def is_org_token(tok):
    if tok.upos == "PROPN":
        return True
    if tok.upos in {"X", "SYM"}:
        return True
    if tok.form in {"&", "-", "–", "'", "’", "\"", "."}:
        return True
    if re.fullmatch(r"\d+(/\d+)*", tok.form):
        return True
    return False

def contains_org_keyword(span):
    for tok in span:
        if tok.form in ORG_KEYWORDS or tok.lemma in ORG_KEYWORDS:
            return True
    return False

def detect_tags_for_sentence(sent):
    n = len(sent)
    tags = ["O"] * n

    # LEG entities
    i = 0
    while i < n:
        t = sent[i]
        if legal_start(t.form, t.lemma, t.upos):
            tags[i] = "B-LEG"
            j = i + 1
            while j < n and legal_continuation(sent[j].form, sent[j].lemma, sent[j].upos):
                tags[j] = "I-LEG"
                j += 1
            i = j
        else:
            i += 1

    # MON entities - currency patterns
    for i, t in enumerate(sent):
        if is_currency(t.form):
            if tags[i].startswith("B-LEG") or tags[i].startswith("I-LEG"):
                continue
            start = i
            end = i
            if i > 0 and is_numeric_token(sent[i - 1]):
                start = i - 1
            elif i + 1 < n and is_numeric_token(sent[i + 1]):
                end = i + 1
            # include magnitude words
            if end + 1 < n and (
                sent[end + 1].lemma in {"Million", "Milliarde", "tausend"}
                or sent[end + 1].form in {"Mio.", "Mrd.", "Tsd."}
            ):
                end = end + 1
            tags[start] = "B-MON"
            for k in range(start + 1, end + 1):
                tags[k] = "I-MON"

    # MON entities - percentages
    for i, t in enumerate(sent):
        if percent_like(t.form, t.lemma):
            if t.form in {"%", "%.", "%,"}:
                if i > 0 and is_numeric_token(sent[i - 1]):
                    if tags[i - 1] == "O":
                        tags[i - 1] = "B-MON"
                    if tags[i] == "O":
                        tags[i] = "I-MON"
            else:
                if t.lemma.lower() == "prozent":
                    if i > 0 and is_numeric_token(sent[i - 1]):
                        if tags[i - 1] == "O":
                            tags[i - 1] = "B-MON"
                        if tags[i] == "O":
                            tags[i] = "I-MON"
                else:
                    if "%" in t.form and not is_numeric_token(t):
                        # pure % sign token; already handled
                        pass
                    else:
                        if tags[i] == "O":
                            tags[i] = "B-MON"

    # ORG entities
    i = 0
    while i < n:
        t = sent[i]
        if is_org_token(t):
            span = []
            j = i
            while j < n and is_org_token(sent[j]):
                if sent[j].form in {",", ";", "/"}:
                    break
                span.append(sent[j])
                j += 1
            if span and contains_org_keyword(span):
                indices = list(range(i, j))
                first_idx = None
                for idx in indices:
                    if sent[idx].upos == "PROPN" or (
                        sent[idx].upos in {"X", "SYM"} and sent[idx].form not in {"\"", "'", "’", "(", ")"}
                    ):
                        first_idx = idx
                        break
                if first_idx is not None:
                    if tags[first_idx] == "O":
                        tags[first_idx] = "B-ORG"
                    for idx in indices:
                        if idx == first_idx:
                            continue
                        if tags[idx] == "O":
                            tags[idx] = "I-ORG"
                i = j
            else:
                i = j
        else:
            i += 1

    return tags

# Write labeled TSV
with open(output_path, "w", encoding="utf-8") as out:
    out.write("sent_id\ttoken_id\ttoken\tner_tag\n")
    for sid, sent in sents:
        tags = detect_tags_for_sentence(sent)
        for tok, tag in zip(sent, tags):
            out.write(f"{sid}\t{tok.id}\t{tok.form}\t{tag}\n")

output_path
