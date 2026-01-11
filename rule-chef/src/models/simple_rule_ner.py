import re
from typing import List, Dict, Tuple

def bio_to_spans(bio: List[str], offsets: List[Tuple[int, int]], text: str) -> List[Dict]:
    spans = []
    n = len(bio)
    i = 0
    while i < n:
        tag = bio[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start = offsets[i][0]
            end = offsets[i][1]
            j = i + 1
            while j < n and bio[j] == f"I-{label}":
                end = offsets[j][1]
                j += 1
            spans.append({"label": label, "start": start, "end": end, "text": text[start:end]})
            i = j
        else:
            i += 1
    return spans


class SimpleRuleNER:
    def __init__(self):
        # allow negatives, thousand separators, optional decimals, percentages
        self.mon_number_re = re.compile(
            r"^-?\d{1,3}([.,]\d{3})+([.,]\d+)?$"   
            r"|^-?\d+,\d+$"                      
            r"|^-?\d+(,\d+)?%$"                    
            r"|^-?\d{3,}$"                        
        )
        self.currency_tokens = {"€", "eur", "eur.", "euro", "euro."}

        self.leg_starters = {"§", "Artikel", "Artikel.", "Art.", "Art", "Artikels"}
        self.leg_follow = {"Absatz", "Abs.", "Satz", "Nr.", "Nr", "Nummer", "(", ")", "-", "WpHG"}
        self.leg_follow_lower = {x.lower() for x in self.leg_follow}

        self.org_keywords = {"bank", "bundesbank", "sparkasse", "genossenschaftsbank"}
        self.org_suffixes = {"ag", "gmbh", "kg", "kgaa", "se", "plc", "ltd", "llc", "inc.", "sarl"}

    def fit(self, train_texts, train_spans):
        return self  # no training

    def _is_numeric_like(self, tok: str) -> bool:
        return any(ch.isdigit() for ch in tok)

    def predict_sentence(self, tokens: List[str]) -> List[str]:
        n = len(tokens)
        labels = ["O"] * n
        lower = [t.lower() for t in tokens]

        # ===== LEG =====
        i = 0
        while i < n:
            t = tokens[i]
            tl = lower[i]

            if t == "§":
                labels[i] = "B-LEG"
                j = i + 1
                while j < n and tokens[j] not in {".", ";"} and j < i + 8:
                    if labels[j] == "O":
                        labels[j] = "I-LEG"
                    j += 1
                i = j
                continue

            if t in {"Artikel", "Artikel.", "Art.", "Art", "Artikels"}:
                labels[i] = "B-LEG"
                j = i + 1
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    # FIX: use tj, not t
                    if self._is_numeric_like(tj) or (tj in self.leg_follow) or (tlj in self.leg_follow_lower):
                        if labels[j] == "O":
                            labels[j] = "I-LEG"
                        j += 1
                    else:
                        break
                i = j
                continue

            i += 1

        # ===== MON =====
        for i, t in enumerate(tokens):
            if labels[i] != "O":
                continue
            tl = lower[i]

            if self.mon_number_re.match(t) or t.endswith("%"):
                labels[i] = "B-MON"
                j = i + 1
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    if tj.endswith("%") or (tlj in self.currency_tokens) or self.mon_number_re.match(tj):
                        if labels[j] == "O":
                            labels[j] = "I-MON"
                        j += 1
                    else:
                        break
                continue

            if tl in self.currency_tokens and i > 0 and self._is_numeric_like(tokens[i - 1]):
                if labels[i - 1] == "O":
                    labels[i - 1] = "B-MON"
                labels[i] = "I-MON"

        # ===== ORG =====
        for i, t in enumerate(tokens):
            if labels[i] != "O":
                continue
            tl = lower[i]

            if tl in self.org_keywords:
                labels[i] = "B-ORG"
                j = i + 1
                while j < n:
                    tj = tokens[j]
                    tlj = lower[j]
                    if (
                        (len(tj) > 0 and tj[0].isupper())
                        or (tlj in self.org_keywords)
                        or (tlj in self.org_suffixes)
                        or (tj in {"-", "–", "&", "’s", "'s", ",", "."})
                    ):
                        if labels[j] == "O":
                            labels[j] = "I-ORG"
                        j += 1
                    else:
                        break
                continue

            if tl in self.org_suffixes:
                start = i
                while start - 1 >= 0 and len(tokens[start - 1]) > 0 and tokens[start - 1][0].isupper() and labels[start - 1] == "O":
                    start -= 1
                labels[start] = "B-ORG"
                for j in range(start + 1, i + 1):
                    if labels[j] == "O":
                        labels[j] = "I-ORG"

        return labels


    def predict_item(self, item: Dict) -> List[Dict]:
        tokens = item["tokens"]
        offsets = item["offsets"]
        text = item["text"]
        bio = self.predict_sentence(tokens)
        return bio_to_spans(bio, offsets, text)
