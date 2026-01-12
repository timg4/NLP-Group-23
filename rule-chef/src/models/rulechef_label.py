import json
import os
import random
from typing import Dict, List, Tuple

from openai import OpenAI
from rulechef import Task, RuleChef, RuleFormat
from rulechef.core import Rule, RuleFormat as RF  # Rule dataclass lives here

SEED = 1337


def _rule_to_dict(r: Rule) -> Dict:
    d = r.to_dict()
    # RuleFormat is stored as string in to_dict() already
    return d


def _rule_from_dict(d: Dict) -> Rule:
    # tolerate minor key issues
    fmt = d.get("format") or d.get("Format")
    if isinstance(fmt, str):
        fmt = RF(fmt.lower())
    return Rule(
        id=d.get("id", ""),
        name=d.get("name", ""),
        description=d.get("description", ""),
        format=fmt,
        content=d.get("content") or d.get("Content") or "",
        priority=int(d.get("priority") or d.get("Priority") or 1),
        confidence=float(d.get("confidence") or 0.5),
    )


class RuleChefLabelExtractor:
    def __init__(
        self,
        label: str,
        model: str = "gpt-3.5-turbo-1106",
        cache_dir: str = "./models/rulechef",
        seed: int = SEED,
        max_train_items: int = 80,
    ):
        self.label = label
        self.model = model
        self.cache_dir = cache_dir
        self.rng = random.Random(seed)
        self.max_train_items = max_train_items

        os.makedirs(self.cache_dir, exist_ok=True)
        self.chef = None

    def _task(self) -> Task:
        if self.label == "MON":
            desc = (
                "Extract MONETARY spans from the input context.\n"
                "Include: percentages (e.g., '35 %', '2,5 %', '100,00 %'), "
                "currency amounts/tokens (e.g., 'EUR', 'Euro', 'Mio. Euro', '€'), "
                "financial amounts with thousand/decimal separators.\n"
                "Exclude: page numbers, section numbers, dates, pure numbering without monetary meaning.\n"
                "Return character spans with start/end offsets into the original context."
            )
        elif self.label == "ORG":
            desc = (
                "Extract ORGANIZATION spans from the input context.\n"
                "Include: company/bank/agency names, acronyms used as organizations "
                "(e.g., 'ISDA', 'OGAW', 'OGA', 'Moody’s', 'MünchenerHyp'), "
                "and legal-form suffixes (AG, GmbH, SE, Ltd, etc.).\n"
                "Exclude: generic words like 'Bank' when it is not a specific name; "
                "exclude currency codes like 'EUR'.\n"
                "Return character spans with start/end offsets into the original context."
            )
        else:  # LEG
            desc = (
                "Extract LEGAL_REFERENCE spans from the input context.\n"
                "Include: § references, 'Artikel ... Absatz ...', 'Nr.', 'Abs.' etc "
                "as part of a legal citation.\n"
                "Exclude: standalone 'Artikel' or 'Absatz' without a real citation.\n"
                "Return character spans with start/end offsets into the original context."
            )

        return Task(
            name=f"{self.label} span extraction",
            description=desc,
            input_schema={"context": "string"},
            output_schema={"spans": "List[Span]"},
        )

    def _build_train_items(self, gold_by_key: Dict) -> List[Tuple[str, List[Dict]]]:
        items = []
        for _, item in gold_by_key.items():
            text = item["text"]
            spans = [sp for sp in item["gold_spans"] if sp["label"] == self.label]
            out_spans = []
            for sp in spans:
                out_spans.append({
                    "text": sp.get("text", text[sp["start"]:sp["end"]]),
                    "start": sp["start"],
                    "end": sp["end"],
                })
            items.append((text, out_spans))

        # downsample to keep cost under control
        self.rng.shuffle(items)
        return items[: self.max_train_items]

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"rules_{self.label}.json")

    def fit(self, gold_by_key: Dict):
        train_items = self._build_train_items(gold_by_key)

        chef = RuleChef(
            task=self._task(),
            client=OpenAI(),
            model=self.model,
            allowed_formats=[RuleFormat.CODE],     
            allow_llm_fallback=False,             
        )

        for text, spans in train_items:
            chef.add_example(
                input_data={"context": text},
                output_data={"spans": spans},
            )

        rules, metrics = chef.learn_rules()
        # persist
        payload = {
            "label": self.label,
            "model": self.model,
            "rules": [_rule_to_dict(r) for r in rules],
            "metrics": metrics,
        }
        with open(self._cache_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.chef = chef
        return metrics

    def load(self):
        path = self._cache_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing rules file: {path}. Run fit() first.")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        rules = [_rule_from_dict(d) for d in payload["rules"]]

        chef = RuleChef(
            task=self._task(),
            client=OpenAI(),
            model=self.model,
            allowed_formats=[RuleFormat.CODE],
            allow_llm_fallback=False,
        )
        chef.dataset.rules = rules
        self.chef = chef

    def predict(self, text: str) -> List[Dict]:
        if self.chef is None:
            self.load()

        out = self.chef.extract({"context": text})
        spans = out.get("spans", [])

        pred = []
        for sp in spans:
            # Span object has .start/.end/.text, but tolerate dict too
            start = int(getattr(sp, "start", sp["start"]))
            end = int(getattr(sp, "end", sp["end"]))
            sp_text = getattr(sp, "text", None) or text[start:end]

            pred.append({
                "label": self.label,
                "start": start,
                "end": end,
                "text": sp_text,
            })
        return pred