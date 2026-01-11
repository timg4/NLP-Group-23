import re
from typing import Dict, List


class GazetteerNER:
    def __init__(self, labels=None, min_len=3):
        self.labels = labels or ["ORG", "MON", "LEG"]
        self.min_len = min_len
        self.phrases: Dict[str, List[str]] = {lab: [] for lab in self.labels}

    def fit(self, train_texts: List[str], train_spans: List[List[Dict]]):
        #collect unique surface forms per label
        tmp = {lab: set() for lab in self.labels}
        for text, spans in zip(train_texts, train_spans):
            for sp in spans:
                lab = sp["label"]
                if lab not in tmp:
                    continue
                surface = text[sp["start"]:sp["end"]].strip()
                if len(surface) >= self.min_len:
                    tmp[lab].add(surface)

        #longest-first helps reduce overlap problems a bit
        for lab in self.labels:
            self.phrases[lab] = sorted(tmp[lab], key=len, reverse=True)

    def predict(self, text: str) -> List[Dict]:
        candidates = []

        for lab in self.labels:
            for phrase in self.phrases.get(lab, []):
                # literal match, case-insensitive
                pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
                for m in pattern.finditer(text):
                    candidates.append({
                        "start": m.start(),
                        "end": m.end(),
                        "label": lab
                    })

        #Resolve overlaps: greedy by longest span first
        candidates.sort(key=lambda s: (-(s["end"] - s["start"]), s["start"]))
        chosen = []
        occupied = []  #list of (start,end)

        for sp in candidates:
            s, e = sp["start"], sp["end"]
            overlap = any(not (e <= os or s >= oe) for os, oe in occupied)
            if overlap:
                continue
            chosen.append(sp)
            occupied.append((s, e))

        #sort back for sanity
        chosen.sort(key=lambda s: (s["start"], s["end"]))
        return chosen