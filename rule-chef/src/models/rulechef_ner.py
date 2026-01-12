from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI
from rulechef import RuleChef, Task
from rulechef.core import RuleFormat
from rulechef.core import Rule, RuleFormat
import uuid


LABELS = ["ORG", "MON", "LEG"]


def _task_for(label: str) -> Task:
    if label == "MON":
        desc = (
            "Extract ONLY MON (monetary/financial amount) spans from German financial text.\n"
            "HARD RULES:\n"
            "- A MON span MUST contain at least one digit.\n"
            "- A MON span MUST contain a currency/amount marker OR be a percentage.\n"
            "  Acceptable markers: €, EUR, Euro, USD, CHF, GBP, JPY, Mio., Mrd., Tsd., Prozent, %.\n"
            "- DO NOT extract section numbers, list indices, headings like 'E.3', 'B.2', or plain integers without any marker.\n"
            "- Include the marker in the span (e.g. '100,00 %', '2,5 %', '35 %', '10 Mio. EUR').\n"
            "Return character offsets into the ORIGINAL context.\n"
        )
    elif label == "ORG":
        desc = (
            "Extract ONLY ORG (organizations: companies, banks, authorities, institutions).\n"
            "HARD RULES:\n"
            "- Do NOT extract generic capitalized noun phrases (German capitalizes nouns).\n"
            "- Prefer organizations with legal forms/keywords: GmbH, AG, SE, KG, OHG, KGaA, e.V., Stiftung,\n"
            "  Bank, Sparkasse, Versicherung, Holding, Gruppe, Ministerium, Amt, Behörde.\n"
            "- Abbreviations (2–5 caps) are ORG ONLY if they are known organizations or occur with an org keyword nearby.\n"
            "- Never return a span that starts at character 0 unless it is clearly the org name.\n"
            "Return character offsets into the ORIGINAL context.\n"
        )
    else:  # LEG
        desc = (
            "Extract ONLY LEG (legal references).\n"
            "HARD RULES:\n"
            "- Only extract spans that include legal markers like: §, Art., Artikel, Abs., Satz, Nr., Z., lit.\n"
            "- Also allow law abbreviations ONLY when used as a law reference (e.g., 'BGB', 'HGB', 'AktG', 'DSGVO')\n"
            "  but DO NOT match arbitrary words or full sentences.\n"
            "- DO NOT extract plain dates, plain numbers, or quoted text unless it is a legal reference.\n"
            "Return character offsets into the ORIGINAL context.\n"
        )

    return Task(
        name=f"extract_{label}",
        description=desc,
        input_schema={"context": "string"},
        output_schema={"spans": "List[Span]"},
    )

@dataclass
class MultiRuleChefNER:
    model_name: str = "gpt-3.5-turbo-1106"
    k_pos: int = 12
    k_neg: int = 12
    storage_path: str = "./rulechef_data"
    max_refinement_iterations: int = 2

    # internal
    chefs: Optional[Dict[str, RuleChef]] = None

    def fit(self, train_texts: List[str], train_spans: List[List[Dict]], fold_name: str = "fold") -> "MultiRuleChefNER":
        """
        train_texts: list of sentence texts
        train_spans: list parallel to train_texts; each item is list of gold span dicts with label/start/end/(text)
        fold_name: used to namespace saved datasets per fold
        """
        client = OpenAI()
        self.chefs = {}

        from pathlib import Path

        for lab in LABELS:
            chef = RuleChef(
                task=_task_for(lab),
                client=client,
                dataset_name=f"{fold_name}_{lab}",
                storage_path=self.storage_path,
                allowed_formats=[RuleFormat.REGEX, RuleFormat.CODE],  
                auto_trigger=False,
                model=self.model_name,
                sampling_strategy="balanced",
            )

            # add examples for THIS label only
            for text, spans in zip(train_texts, train_spans):
                lab_spans = [sp for sp in spans if sp["label"] == lab]
                out = {
                    "spans": [
                        {
                            "start": int(sp["start"]),
                            "end": int(sp["end"]),
                            "text": text[int(sp["start"]):int(sp["end"])],
                        }
                        for sp in lab_spans
                    ]
                }
                chef.add_example({"context": text}, out)

        
            chef.add_feedback(f"Only output {lab}. Never output ORG/MON/LEG other than {lab}.")
            chef.add_feedback("Output must be valid JSON. Spans must be character offsets. No markdown.")
            chef.add_feedback("Hard constraint: NEVER propose a generic capitalization-based ORG rule.")
            chef.add_feedback("Hard constraint: For MON, require at least one digit OR a currency symbol like €.")
            chef.add_feedback("Hard constraint: Output strictly valid JSON. No markdown.")
            if lab == "MON":
                chef.add_feedback("Reject any rule that matches plain numbers without % or a currency/amount marker.")
                chef.add_feedback("Always include '%' or currency marker inside the span.")
            elif lab == "ORG":
                chef.add_feedback("Reject any rule that matches 'Capitalized words' in general. That is NOT ORG.")
                chef.add_feedback("ORG must usually contain a legal form/keyword or be a known bank/group name.")
            else:  # LEG
                chef.add_feedback("Reject any rule that matches arbitrary words or sentences. Only legal markers like §/Art./Abs./Nr./Satz.")
            
            chef.learn_rules(run_evaluation=True, max_refinement_iterations=2, sampling_strategy="balanced")
            
            if lab == "MON":
                chef.dataset.rules.insert(0, Rule(
                    id=str(uuid.uuid4())[:8],
                    name="MON percent with space",
                    description="Capture percentages like '100,00 %' or '35 %'",
                    format=RuleFormat.REGEX,
                    content=r"\b\d{1,3}(?:[.,]\d{1,2})?\s?%\b",
                    priority=100,
                    confidence=0.9
                ))
            print(f"\n[{fold_name}] {lab} learned rules:")
            for r in chef.dataset.rules:
                print(" -", r.name, "|", r.format.value, "| priority", r.priority)
                print(r.content[:400])
                print()

            self.chefs[lab] = chef

        return self

    def predict(self, text: str) -> List[Dict]:
        assert self.chefs is not None, "Call fit() first."
        preds: List[Dict] = []

        for lab, chef in self.chefs.items():
            out = chef.learner._apply_rules(chef.dataset.rules, {"context": text})  # must return {"spans":[...]}
            spans = out.get("spans", []) if isinstance(out, dict) else []

            for sp in spans:
                s = int(sp["start"])
                e = int(sp["end"])
                preds.append({
                    "label": lab,
                    "start": s,
                    "end": e,
                    "text": text[s:e],
                })

        # de-dupe exact duplicates
        seen = set()
        uniq = []
        for sp in preds:
            k = (sp["label"], sp["start"], sp["end"])
            if k not in seen:
                uniq.append(sp)
                seen.add(k)
        return uniq
