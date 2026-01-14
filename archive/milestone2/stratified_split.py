"""
Shared stratified 80/20 split helper (seeded) for comparability.
"""
import random
from typing import Dict, List, Tuple

SPLIT_SEED = 2323


def _sentence_stratum(labels: List[str]) -> str:
    if any(lab.endswith("-MON") for lab in labels):
        return "MON"
    if any(lab.endswith("-LEG") for lab in labels):
        return "LEG"
    if any(lab.endswith("-ORG") for lab in labels):
        return "ORG"
    return "NONE"


def stratified_split(
    sentences: List[Dict],
    train_ratio: float = 0.8,
    seed: int = SPLIT_SEED,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    """
    Stratify by sentence-level presence of MON/LEG/ORG, fallback NONE.
    Returns train/dev lists plus per-stratum counts.
    """
    buckets: Dict[str, List[Dict]] = {}
    for sent in sentences:
        key = _sentence_stratum(sent["labels"])
        buckets.setdefault(key, []).append(sent)

    rng = random.Random(seed)
    train_set, dev_set = [], []
    counts: Dict[str, Dict[str, int]] = {}

    for key, bucket in buckets.items():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n_total = len(bucket)
        if n_total == 1:
            n_dev = 1
        else:
            n_dev = max(1, int(round((1 - train_ratio) * n_total)))
        n_dev = min(n_dev, n_total - 1) if n_total > 1 else n_dev
        dev_set.extend(bucket[:n_dev])
        train_set.extend(bucket[n_dev:])
        counts[key] = {"total": n_total, "train": n_total - n_dev, "dev": n_dev}

    rng.shuffle(train_set)
    rng.shuffle(dev_set)
    return train_set, dev_set, counts
