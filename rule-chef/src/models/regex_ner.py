import re
from typing import Dict, List, Tuple
from src.models.base import NERModel

class RegexNER(NERModel):
    def __init__(self):

        self.mon_patterns = [

            re.compile(r"(?:€\s?)(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:,\d+)?)"),

            re.compile(r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:,\d+)?)(?:\s?(?:EUR|Euro))"),

            re.compile(r"(\d+(?:[.,]\d+)?)\s?%"),
        ]

        self.leg_patterns = [

            re.compile(r"§\s*\d+[a-zA-Z]?(?:\s*(?:Abs\.|Absatz)\s*\d+[a-zA-Z]?)?(?:\s*(?:Nr\.|Nummer)\s*\d+)?"),
        
            re.compile(r"Art\.?\s*\d+[a-zA-Z]?(?:\s*(?:Abs\.|Absatz)\s*\d+[a-zA-Z]?)?"),
   
            re.compile(r"(?:sub-)?paragraph\s*\(\s*\d+\s*\)", re.IGNORECASE),
        ]

    def predict(self, text: str) -> List[Dict]:
        spans: List[Dict] = []
        spans.extend(self._find_all(text, self.mon_patterns, "MON"))
        spans.extend(self._find_all(text, self.leg_patterns, "LEG"))


        spans = self._dedup_and_resolve_overlaps(spans)
        return spans

    def _find_all(self, text: str, patterns: List[re.Pattern], label: str) -> List[Dict]:
        out = []
        for pat in patterns:
            for m in pat.finditer(text):
                start, end = m.span(0)
                out.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": text[start:end]
                })
        return out

    def _dedup_and_resolve_overlaps(self, spans: List[Dict]) -> List[Dict]:
        # sort: longer first, then earlier
        spans = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), s["start"], s["label"]))
        kept = []
        occupied: List[Tuple[int, int]] = []

        def overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        for sp in spans:
            interval = (sp["start"], sp["end"])
            if any(overlaps(interval, iv) for iv in occupied):
                continue
            kept.append(sp)
            occupied.append(interval)

        # return sorted by start for readability
        return sorted(kept, key=lambda s: (s["start"], s["end"], s["label"]))