#shows the examples that were missed in fold0
from typing import Dict, List, Tuple

def _key(sp: Dict) -> Tuple[str, int, int]:
    return (sp["label"], sp["start"], sp["end"])

def dump_errors(
    text_by_key: Dict[str, str],
    gold_by_key: Dict[str, List[Dict]],
    pred_by_key: Dict[str, List[Dict]],
    max_items: int = 25,
):
    """
    Prints a small sample of FNs/FPs per label with the covered text.
    """
    labels = ["ORG", "MON", "LEG"]

    # Collect per label errors
    fn = {lab: [] for lab in labels}
    fp = {lab: [] for lab in labels}
   

    for k in gold_by_key:
        text = text_by_key[k]
        gold = gold_by_key[k]
        pred = pred_by_key.get(k, [])

        gset = {_key(sp) for sp in gold}
        pset = {_key(sp) for sp in pred}

        for sp in gold:
            if _key(sp) not in pset:
                span_text = sp.get("text")
                if span_text is None:
                    span_text = text[sp["start"]:sp["end"]]
                fn[sp["label"]].append((k, span_text, text))
                
        for sp in pred:
            if _key(sp) not in gset:
                span_text = sp.get("text")
                if span_text is None:
                    span_text = text[sp["start"]:sp["end"]]
                fp[sp["label"]].append((k, span_text, text))
            
    def _print_block(title: str, items: List[Tuple[str, str, str]]):
        print("\n" + title)
        print("-" * len(title))
        for k, span_text, sent_text in items[:max_items]:
            print(f"[{k}] span='{span_text}'")
            print(f"  sent: {sent_text}")

    for lab in labels:
        _print_block(f"FALSE NEGATIVES ({lab})", fn[lab])
        _print_block(f"FALSE POSITIVES ({lab})", fp[lab])