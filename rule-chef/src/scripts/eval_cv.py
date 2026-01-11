#this loads the gold dataset, loads the folds, trains, fits, predicts and give scores per folds and the mean 
from pydoc import text
from pyexpat import model
from statistics import mean
from typing import Dict, List

from src.data.gold_dataset import load_gold
from src.data.folds import load_folds
from src.eval.span_eval import score_spans
from src.models.empty import EmptyNER
from src.models.regex_ner import RegexNER
from src.models.simple_rule_ner import SimpleRuleNER
from src.eval.error_dump import dump_errors
from src.models.gazetteer_ner import GazetteerNER



def main():
    gold = load_gold("./gold_data/gold.conllu")
    folds = load_folds("./src/data/splits/folds_5.json")

    all_keys = list(gold.keys())

    fold_micro_f1 = []
    fold_label_f1 = {"ORG": [], "MON": [], "LEG": []}

    for fold_name, test_keys in folds.items():
        test_keys = set(test_keys)
        train_keys = [k for k in all_keys if k not in test_keys]

        train_texts = [gold[k]["text"] for k in train_keys]
        train_spans = [gold[k]["gold_spans"] for k in train_keys]

        
        model = GazetteerNER()
        model.fit(train_texts, train_spans)

        gold_all = []
        pred_all = []

        text_by_key = {}
        gold_by_key = {}
        pred_by_key = {}

        for k in test_keys:
            text = gold[k]["text"]
            gold_sp = gold[k]["gold_spans"]
            pred_sp = model.predict(text)

            text_by_key[k] = text
            gold_by_key[k] = gold_sp
            pred_by_key[k] = pred_sp

            gold_all.extend(gold_sp)
            pred_all.extend(pred_sp)

        scores = score_spans(gold_all, pred_all)
        micro_f1 = scores["micro"]["f1"]

        fold_micro_f1.append(micro_f1)
        for lab in ["ORG", "MON", "LEG"]:
            fold_label_f1[lab].append(scores["by_label"][lab]["f1"])

        print(
            f"{fold_name}: micro_f1={micro_f1:.4f} "
            f"ORG={scores['by_label']['ORG']['f1']:.4f} "
            f"MON={scores['by_label']['MON']['f1']:.4f} "
            f"LEG={scores['by_label']['LEG']['f1']:.4f}"
        )

      
        if fold_name == "fold_0":
            dump_errors(text_by_key, gold_by_key, pred_by_key, max_items=15)

    print("\nMEAN over folds:")
    print(f"micro_f1={mean(fold_micro_f1):.4f}")
    for lab in ["ORG", "MON", "LEG"]:
        print(f"{lab}_f1={mean(fold_label_f1[lab]):.4f}")


if __name__ == "__main__":
    main()