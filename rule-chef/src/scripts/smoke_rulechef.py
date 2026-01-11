import random

from openai import OpenAI
from rulechef import Task, RuleChef

from src.data.gold_dataset import load_gold

SEED = 1337

def pick_examples(gold, label: str, k_pos: int = 6):
    ex = []
    for _, item in gold.items():
        spans = [sp for sp in item["gold_spans"] if sp["label"] == label]
        if not spans:
            continue
        out_spans = [
            {
                "text": sp.get("text", item["text"][sp["start"]:sp["end"]]),
                "start": sp["start"],
                "end": sp["end"],
            }
            for sp in spans
        ]
        ex.append((item["text"], out_spans))
    return ex[:k_pos]

def main():
    rng = random.Random(SEED)

    gold = load_gold("./gold_data/gold.conllu")

    label = "MON"  # start with ONE label
    examples = pick_examples(gold, label=label, k_pos=6)
    if len(examples) < 3:
        raise RuntimeError(f"Not enough positive examples for {label} to smoke test.")

    question = f"Extract all {label} spans from the context. Return character spans."

    task = Task(
        name=f"{label} span extraction",
        description=(
            f"Extract {label} entity spans from the context. "
            "Return character-based spans with start/end offsets into the original context."
        ),
        input_schema={"question": "string", "context": "string"},
        output_schema={"spans": "List[Span]"},
    )

    chef = RuleChef(task=task, client=OpenAI())

    # add a few training examples
    for text, spans in examples:
        chef.add_example(
            input_data={"question": question, "context": text},
            output_data={"spans": spans},
        )

    rules, metrics = chef.learn_rules()
    print("\n=== LEARNED RULES ===")
    print(rules, metrics)

    # test on a random example
    test_text, gold_spans = rng.choice(examples)
    print("\n=== TEST TEXT ===")
    print(test_text)

    pred = chef.extract({"question": question, "context": test_text})

    print("\n=== PREDICTION ===")
    print(pred)

    print("\n=== GOLD (for sanity) ===")
    print(gold_spans)

if __name__ == "__main__":
    main()
