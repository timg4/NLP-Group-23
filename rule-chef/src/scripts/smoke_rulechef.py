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
        out_spans = [{"text": sp.get("text", item["text"][sp["start"]:sp["end"]]),
                      "start": sp["start"], "end": sp["end"]} for sp in spans]
        ex.append((item["text"], out_spans))
    return ex[:k_pos]

def main():
    rng = random.Random(SEED)

    gold = load_gold("./gold_data/gold.conllu")

    label = "MON"  # start with ONE label
    examples = pick_examples(gold, label=label, k_pos=6)
    if len(examples) < 3:
        raise RuntimeError(f"Not enough positive examples for {label} to smoke test.")

    task = Task(
        name=f"{label} span extraction",
        description=(
            f"Extract {label} entity spans from the input text. "
            "Return character-based spans with start/end offsets into the original text."
        ),
        input_schema={"text": "string"},
        output_schema={"spans": "List[Span]"},
    )

    chef = RuleChef(task=task, client=OpenAI())

    # add a few training examples
    for text, spans in examples:
        chef.add_example(
            input_data={"text": text},
            output_data={"spans": spans},
        )

    # learn rules (RuleChef README shows learn_rules() without args) :contentReference[oaicite:2]{index=2}
    rules = chef.learn_rules()
    print("\n=== LEARNED RULES ===")
    print(rules)

    # test on a random example text (could also pick a different one)
    test_text, gold_spans = rng.choice(examples)
    print("\n=== TEST TEXT ===")
    print(test_text)

    # extract: try kwargs first, fallback to positional (RuleChef matches schema fields)
    try:
        pred = chef.extract(text=test_text)
    except TypeError:
        pred = chef.extract(test_text)

    print("\n=== PREDICTION ===")
    print(pred)

    print("\n=== GOLD (for sanity) ===")
    print(gold_spans)

if __name__ == "__main__":
    main()