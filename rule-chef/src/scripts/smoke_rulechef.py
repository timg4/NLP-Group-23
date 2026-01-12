import random
from openai import OpenAI
from rulechef import Task, RuleChef

from src.data.gold_dataset import load_gold
from src.models.rulechef_ner import _task_for

SEED = 1337
LABELS = ["MON", "ORG", "LEG"]

def pick_examples(gold, label: str, k_pos: int = 8, k_neg: int = 8, seed: int = SEED):
    rng = random.Random(seed)

    pos = []
    neg = []

    items = list(gold.items())
    rng.shuffle(items)

    for _, item in items:
        spans = [sp for sp in item["gold_spans"] if sp["label"] == label]

        if spans:
            out_spans = [
                {
                    "text": sp.get("text", item["text"][sp["start"]:sp["end"]]),
                    "start": int(sp["start"]),
                    "end": int(sp["end"]),
                }
                for sp in spans
            ]
            pos.append((item["text"], out_spans))
        else:
            neg.append((item["text"], []))

        if len(pos) >= k_pos and len(neg) >= k_neg:
            break

    return pos[:k_pos], neg[:k_neg]

def train_one_label(gold, label: str):
    pos, neg = pick_examples(gold, label=label, k_pos=8, k_neg=8)

    if len(pos) < 3:
        raise RuntimeError(f"Not enough positive examples for {label} (need >= 3). Got {len(pos)}.")

    task = task = _task_for(label)

    chef = RuleChef(
        task=task,
        client=OpenAI(),
        dataset_name=f"smoke_{label}",  # IMPORTANT: don't mix datasets across labels
    )

    # Add positives
    for text, spans in pos:
        chef.add_example(
            input_data={"context": text},
            output_data={"spans": spans},
        )

    # Add negatives (crucial)
    for text, spans in neg:
        chef.add_example(
            input_data={"context": text},
            output_data={"spans": spans},
        )

    rules = chef.learn_rules(run_evaluation=True, max_refinement_iterations=1, sampling_strategy="balanced")
    return chef, rules, pos, neg

def main():
    rng = random.Random(SEED)
    gold = load_gold("./gold_data/gold.conllu")

    for label in LABELS:
        print("\n" + "=" * 70)
        print(f"TRAINING RULECHEF FOR: {label}")
        print("=" * 70)

        chef, rules, pos, neg = train_one_label(gold, label)
        print("\n=== LEARNED RULES ===")
        print(rules)

        # Test one positive + one negative
        test_pos_text, gold_spans = rng.choice(pos)
        pred_pos = chef.extract({"context": test_pos_text})

        test_neg_text, _ = rng.choice(neg)
        pred_neg = chef.extract({"context": test_neg_text})

        print("\n--- POSITIVE TEST ---")
        print(test_pos_text)
        print("PRED:", pred_pos)
        print("GOLD:", gold_spans)

        print("\n--- NEGATIVE TEST ---")
        print(test_neg_text)
        print("PRED:", pred_neg)

if __name__ == "__main__":
    main()
