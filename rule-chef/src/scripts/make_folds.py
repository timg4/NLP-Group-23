# src/scripts/make_folds.py
import json
import random
from collections import defaultdict

from src.data.conllu_reader import read_conllu
from src.data.gold_to_spans import align_tokens_to_text, bio_to_spans

LABELS = ["ORG", "MON", "LEG"]
N_FOLDS = 5
SEED = 1337

def signature(present: dict) -> str:
    # ORG, MON, LEG order
    return "".join("1" if present[l] else "0" for l in LABELS)

def main():
    rng = random.Random(SEED)
    path = "./gold_data/gold.conllu"  # adjust if needed

    groups = defaultdict(list)

    for sent in read_conllu(path):
        char_tokens = align_tokens_to_text(sent)
        spans = bio_to_spans(char_tokens)

        present = {lab: False for lab in LABELS}
        for sp in spans:
            present[sp["label"]] = True

        key = f"{sent.doc_id}:{sent.sent_id}" if sent.doc_id is not None else str(sent.sent_id)
        groups[signature(present)].append(key)

    for sig in groups:
        rng.shuffle(groups[sig])


    folds = [[] for _ in range(N_FOLDS)]
    for sig, keys in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        for i, k in enumerate(keys):
            folds[i % N_FOLDS].append(k)


    total = sum(len(f) for f in folds)
    if total % N_FOLDS != 0:
        raise ValueError(f"Total sentences {total} not divisible by {N_FOLDS}.")
    target = total // N_FOLDS  #

    all_000 = set(groups.get("000", []))

    def fold_size(i: int) -> int:
        return len(folds[i])


    while True:
        big = max(range(N_FOLDS), key=fold_size)
        small = min(range(N_FOLDS), key=fold_size)

        if fold_size(big) == target and fold_size(small) == target:
            break
        if fold_size(big) <= target or fold_size(small) >= target:
            break  

        movable = [k for k in folds[big] if k in all_000]
        if not movable:
            break  

        k = movable[-1]  
        folds[big].remove(k)
        folds[small].append(k)

    # 4) Print summaries
    fold_sig_counts = [defaultdict(int) for _ in range(N_FOLDS)]
    group_sets = {sig: set(keys) for sig, keys in groups.items()}

    print("Signature legend (ORG,MON,LEG):")
    print(" 100=ORG, 010=MON, 001=LEG, 110=ORG+MON, 101=ORG+LEG, 011=MON+LEG, 111=all, 000=none")
    for i in range(N_FOLDS):
        for sig, s in group_sets.items():
            fold_sig_counts[i][sig] = sum(1 for k in folds[i] if k in s)
        print(f"fold_{i}: n={len(folds[i])}, sig_counts={dict(fold_sig_counts[i])}")

    # 5) Write out
    out = {f"fold_{i}": sorted(folds[i]) for i in range(N_FOLDS)}
    with open("./src/data/splits/folds_5.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
