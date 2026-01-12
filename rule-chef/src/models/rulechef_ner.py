from __future__ import annotations
import random
from typing import List, Dict, Any, Optional

from openai import OpenAI
from rulechef import Task, RuleChef

class RuleChefNER:
    def __init__(
        self,
        label: str,
        model_name: str = "gpt-3.5-turbo-1106", 
        k_pos: int = 12,
        k_neg: int = 12,
        run_evaluation: bool = True,
        max_refinement_iterations: int = 1,
    ):
        self.label = label
        self.model_name = model_name
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.run_evaluation = run_evaluation
        self.max_refinement_iterations = max_refinement_iterations

        self._chef: Optional[RuleChef] = None

    def fit(self, train_texts: List[str], train_spans: List[List[Dict[str, Any]]]):
        rng = random.Random(6069)

        pos = []
        neg = []
        for text, spans in zip(train_texts, train_spans):
            these = [sp for sp in spans if sp["label"] == self.label]
            if these:
                out_spans = []
                for sp in these:
                    out_spans.append({
                        "text": sp.get("text", text[sp["start"]:sp["end"]]),
                        "start": int(sp["start"]),
                        "end": int(sp["end"]),
                    })
                pos.append((text, out_spans))
            else:
                neg.append((text, []))

        rng.shuffle(pos)
        rng.shuffle(neg)
        pos = pos[: self.k_pos]
        neg = neg[: self.k_neg]

        if len(pos) < 3:
            raise RuntimeError(f"Not enough positive training examples for label={self.label}")

        task = Task(
            name=f"{self.label} span extraction",
            description=(
                f"Extract {self.label} entity spans from the input text. "
                "Return character-based spans with start/end offsets."
            ),
            input_schema={"text": "string"},
            output_schema={"spans": "List[Span]"},
        )

        self._chef = RuleChef(
            task=task,
            client=OpenAI(),
            model=self.model_name
        )

        for text, spans in (pos + neg):
            self._chef.add_example(
                input_data={"text": text},
                output_data={"spans": spans},
            )

        # learn rules on TRAIN ONLY
        self._chef.learn_rules(
            run_evaluation=self.run_evaluation,
            max_refinement_iterations=self.max_refinement_iterations,
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if self._chef is None:
            raise RuntimeError("Call fit() before predict().")

        out = self._chef.learner._apply_rules(self._chef.dataset.rules, {"text": text})

        spans = out.get("spans", []) if isinstance(out, dict) else []
        preds = []
        for sp in spans:
            start = int(sp["start"])
            end = int(sp["end"])
            preds.append({
                "label": self.label,
                "start": start,
                "end": end,
                "text": text[start:end],
            })
        return preds
