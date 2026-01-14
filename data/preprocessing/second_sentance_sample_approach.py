#!/usr/bin/env python3
"""
Sample Sentences for Manual Annotation (Improved)
-------------------------------------------------
Randomly sample sentences from a (potentially non-standard) CoNLL-U corpus for manual labeling.
Adds quality filters to avoid fragments, table lines, numeric-only sentences, and very short samples.

Usage:
  python data/sample_for_manual_annotation.py \
    --input data/processed/fincorpus_processed.conllu \
    --output data/manual_annotation \
    --num_sentences 150

Recommended knobs (example):
  --min_tokens 8 --max_tokens 40 --entity_share 0.75 --require_end_punct
"""

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict


ORG_INDICATORS = [
    "Bank", "AG", "GmbH", "SE", "KG", "KGaA", "LLC", "Ltd", "PLC",
    "Landesbank", "Sparkasse", "Versicherung", "Holding", "Konzern"
]
LEGAL_INDICATORS = ["§", "Art.", "Abs.", "Nr.", "Satz", "Ziff.", "lit.", "VO", "RL"]
MONEY_INDICATORS = ["EUR", "€", "USD", "$", "CHF", "GBP", "JPY", "%", "Basispunkte", "bp", "Bps"]
FIN_ABBREV = ["Mio.", "Mrd.", "Tsd.", "p.a.", "YoY", "QoQ"]


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class Token:
    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str


@dataclass
class Sentence:
    doc_id: str
    tokens: List[Token] = field(default_factory=list)

    def add_token_from_fields(self, fields: List[str]) -> None:
        # Expect 10 fields (standard CoNLL-U) with a numeric ID (no ranges, no decimals)
        if len(fields) < 10:
            return
        tid = fields[0]
        if "-" in tid or "." in tid:
            return
        self.tokens.append(Token(*fields[:10]))

    def num_tokens(self) -> int:
        return len(self.tokens)

    def get_text(self) -> str:
        text = ""
        for t in self.tokens:
            text += t.form
            if "SpaceAfter=No" not in (t.misc or ""):
                text += " "
        return text.strip()

    def upos_counts(self) -> Counter:
        return Counter(t.upos for t in self.tokens)

    def ratios(self) -> Dict[str, float]:
        n = max(self.num_tokens(), 1)
        c = self.upos_counts()
        return {
            "num": c.get("NUM", 0) / n,
            "punct": c.get("PUNCT", 0) / n,
            "sym": c.get("SYM", 0) / n,
            "propn": c.get("PROPN", 0) / n,
        }

    def has_verb(self) -> bool:
        return any(t.upos in {"VERB", "AUX"} for t in self.tokens)

    def alpha_token_ratio(self) -> float:
        # Token counts as "alpha" if it contains at least one letter.
        n = max(self.num_tokens(), 1)
        alpha = 0
        for t in self.tokens:
            if re.search(r"[A-Za-zÄÖÜäöüß]", t.form or ""):
                alpha += 1
        return alpha / n

    def looks_like_table_line(self) -> bool:
        # Heuristic: lots of separators or repeated punctuation, or many tokens are numeric/symbolic.
        txt = self.get_text()
        if "|" in txt or "—" in txt or "–" in txt:
            return True
        if re.search(r"_{2,}|-{3,}|={3,}", txt):
            return True
        r = self.ratios()
        if (r["num"] + r["sym"]) > 0.60:
            return True
        return False

    def ends_with_sentence_punct(self) -> bool:
        txt = self.get_text().rstrip()
        return bool(re.search(r"[.!?]([\"'\)\]]+)?$", txt))

    def has_potential_entities(self) -> bool:
        txt = self.get_text()

        # Strong markers
        for ind in ORG_INDICATORS + LEGAL_INDICATORS + MONEY_INDICATORS + FIN_ABBREV:
            if ind in txt:
                return True

        # Proper noun sequences
        propn_runs = 0
        current = 0
        for t in self.tokens:
            if t.upos == "PROPN":
                current += 1
                propn_runs = max(propn_runs, current)
            else:
                current = 0
        if propn_runs >= 2:
            return True

        # All-caps / typical ticker-like tokens (avoid single letters)
        for t in self.tokens:
            f = t.form or ""
            if len(f) >= 2 and f.isupper() and re.search(r"[A-Z]", f):
                return True

        # Numeric amounts with separators
        if re.search(r"\b\d{1,3}(\.\d{3})*(,\d+)?\b", txt):  # German style
            return True
        if re.search(r"\b\d+(\.\d+)?\s*(%|bp|Bps)\b", txt):
            return True

        return False


def load_sentences(file_path: Path, max_sentences: Optional[int] = None) -> List[Sentence]:
    """
    Load sentences from:
      1) Standard CoNLL-U: token lines are TAB-separated with 10 columns, sentences separated by blank lines
      2) Non-standard vertical format: token ID on its own line, followed by 9 lines (form..misc)

    Sentence boundary triggers:
      - blank line
      - token_id == "1" (start of a new sentence)
      - '# newdoc id = ...' lines
    """
    sentences: List[Sentence] = []
    doc_id: str = "UNKNOWN"
    current: Optional[Sentence] = None

    def flush():
        nonlocal current
        if current and current.tokens:
            sentences.append(current)
        current = None

    print(f"Loading sentences from {file_path}...")

    lines = file_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        if not line.strip():
            # sentence boundary in standard CoNLL-U
            flush()
            if max_sentences and len(sentences) >= max_sentences:
                break
            i += 1
            continue

        if line.startswith("# newdoc"):
            # save any pending sentence and start a new document
            flush()
            # allow both "# newdoc id = X" and other variants
            if "=" in line:
                doc_id = line.split("=", 1)[-1].strip()
            else:
                doc_id = line.strip()
            i += 1
            continue

        if line.startswith("#"):
            # ignore other comments; could be extended if needed
            i += 1
            continue

        # Ensure we have a current sentence object
        if current is None:
            current = Sentence(doc_id=doc_id)

        # Case A: Standard CoNLL-U token line (tabs)
        if "\t" in line:
            fields = line.split("\t")
            tid = fields[0]
            # if a new sentence starts (id=1) but we didn't see a blank line (corrupt formatting)
            if tid == "1" and current.tokens:
                flush()
                current = Sentence(doc_id=doc_id)
            current.add_token_from_fields(fields)
            i += 1
            continue

        # Case B: Vertical format: token_id on one line, next 9 lines are the remaining fields
        token_id = line.strip()

        # Need at least 9 following lines for the remaining fields
        if i + 9 >= len(lines):
            break

        fields = [token_id]
        fields.extend(lines[i + 1: i + 10])

        # Move pointer by 10 lines (id + 9 fields)
        i += 10

        # Start of new sentence if token_id == "1"
        if token_id == "1" and current.tokens:
            flush()
            current = Sentence(doc_id=doc_id)

        # Skip multi-word tokens and decimal ids, but still consumed their fields above
        if "-" in token_id or "." in token_id:
            continue

        current.add_token_from_fields(fields)

        if max_sentences and len(sentences) >= max_sentences:
            break

    # flush last
    flush()

    print(f"Loaded {len(sentences)} sentences")
    return sentences


def is_good_candidate(
    s: Sentence,
    min_tokens: int,
    max_tokens: int,
    max_num_ratio: float,
    max_punct_ratio: float,
    max_sym_ratio: float,
    min_alpha_ratio: float,
    require_verb: bool,
    require_end_punct: bool,
    reject_stats: Counter,
) -> bool:
    n = s.num_tokens()
    if n < min_tokens:
        reject_stats["too_short"] += 1
        return False
    if n > max_tokens:
        reject_stats["too_long"] += 1
        return False

    if s.looks_like_table_line():
        reject_stats["table_like"] += 1
        return False

    r = s.ratios()
    if r["num"] > max_num_ratio:
        reject_stats["too_numeric"] += 1
        return False
    if r["punct"] > max_punct_ratio:
        reject_stats["too_punct"] += 1
        return False
    if r["sym"] > max_sym_ratio:
        reject_stats["too_symbolic"] += 1
        return False

    ar = s.alpha_token_ratio()
    if ar < min_alpha_ratio:
        reject_stats["too_few_alpha_tokens"] += 1
        return False

    if require_verb and not s.has_verb():
        reject_stats["no_verb"] += 1
        return False

    if require_end_punct and not s.ends_with_sentence_punct():
        reject_stats["no_end_punct"] += 1
        return False

    reject_stats["kept"] += 1
    return True


def allocate_by_availability(avail: Dict[str, int], total: int) -> Dict[str, int]:
    """
    Allocate 'total' samples over bins proportionally to availability,
    then distribute remainder by largest fractional parts.
    """
    if total <= 0:
        return {k: 0 for k in avail}

    sum_avail = sum(avail.values())
    if sum_avail == 0:
        return {k: 0 for k in avail}

    raw = {k: total * (v / sum_avail) for k, v in avail.items()}
    base = {k: int(raw[k]) for k in raw}
    used = sum(base.values())

    # distribute remainder
    remainder = total - used
    frac = sorted(((raw[k] - base[k], k) for k in raw), reverse=True)
    idx = 0
    while remainder > 0 and idx < len(frac):
        _, k = frac[idx]
        base[k] += 1
        remainder -= 1
        idx += 1

    # cap by availability
    for k in base:
        if base[k] > avail[k]:
            base[k] = avail[k]
    return base


def stratified_sample(
    sentences: List[Sentence],
    num_samples: int,
    seed: int,
    entity_share: float,
    len_short_max: int,
    len_medium_max: int,
) -> List[Sentence]:
    random.seed(seed)

    def len_bin(s: Sentence) -> str:
        n = s.num_tokens()
        if n <= len_short_max:
            return "short"
        elif n <= len_medium_max:
            return "medium"
        return "long"

    entity = []
    non_entity = []
    for s in sentences:
        (entity if s.has_potential_entities() else non_entity).append(s)

    target_entity = int(round(num_samples * entity_share))
    target_non = num_samples - target_entity

    # bucketize
    buckets_entity = defaultdict(list)
    buckets_non = defaultdict(list)
    for s in entity:
        buckets_entity[len_bin(s)].append(s)
    for s in non_entity:
        buckets_non[len_bin(s)].append(s)

    # allocate per bin by availability
    avail_e = {k: len(v) for k, v in buckets_entity.items()}
    avail_n = {k: len(v) for k, v in buckets_non.items()}

    # ensure bins exist
    for b in ("short", "medium", "long"):
        avail_e.setdefault(b, 0)
        avail_n.setdefault(b, 0)
        buckets_entity.setdefault(b, [])
        buckets_non.setdefault(b, [])

    alloc_e = allocate_by_availability(avail_e, target_entity)
    alloc_n = allocate_by_availability(avail_n, target_non)

    sampled: List[Sentence] = []

    def take_from_bucket(bucket: List[Sentence], k: int) -> List[Sentence]:
        if k <= 0:
            return []
        if len(bucket) <= k:
            return bucket[:]
        return random.sample(bucket, k)

    # sample per bin
    for b in ("short", "medium", "long"):
        sampled.extend(take_from_bucket(buckets_entity[b], alloc_e[b]))
        sampled.extend(take_from_bucket(buckets_non[b], alloc_n[b]))

    # fill if short
    if len(sampled) < num_samples:
        remaining = [s for s in sentences if s not in sampled]
        need = num_samples - len(sampled)
        if remaining:
            sampled.extend(random.sample(remaining, min(need, len(remaining))))

    # shuffle
    random.shuffle(sampled)
    return sampled[:num_samples]


def save_as_conllu(sentences: List[Sentence], output_path: Path):
    """Save sentences in CoNLL-U-like format with 11th column for NER tags."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sent_idx, sent in enumerate(sentences, 1):
            f.write(f"# sent_id = {sent_idx}\n")
            f.write(f"# doc_id = {sent.doc_id}\n")
            f.write(f"# text = {sent.get_text()}\n")
            for t in sent.tokens:
                cols = [
                    t.id, t.form, t.lemma, t.upos, t.xpos, t.feats,
                    t.head, t.deprel, t.deps, t.misc,
                    "_"  # NER column
                ]
                f.write("\t".join(cols) + "\n")
            f.write("\n")
    print(f"Saved CoNLL-U format to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample sentences for manual annotation (improved)")
    parser.add_argument("--input", required=True, help="Input CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num_sentences", type=int, default=150, help="Number of sentences to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Quality knobs
    parser.add_argument("--max_sentences_load", type=int, default=50000, help="Max sentences to load (speed)")
    parser.add_argument("--min_tokens", type=int, default=8, help="Minimum tokens per sentence")
    parser.add_argument("--max_tokens", type=int, default=40, help="Maximum tokens per sentence")
    parser.add_argument("--max_num_ratio", type=float, default=0.35, help="Max share of NUM tokens")
    parser.add_argument("--max_punct_ratio", type=float, default=0.30, help="Max share of PUNCT tokens")
    parser.add_argument("--max_sym_ratio", type=float, default=0.20, help="Max share of SYM tokens")
    parser.add_argument("--min_alpha_ratio", type=float, default=0.60, help="Min share of tokens containing letters")
    parser.add_argument("--require_verb", action="store_true", default=True, help="Require VERB/AUX (default on)")
    parser.add_argument("--no_require_verb", action="store_true", help="Disable verb requirement")
    parser.add_argument("--require_end_punct", action="store_true", help="Require sentence-ending punctuation .!?")
    parser.add_argument("--deduplicate", action="store_true", default=True, help="Deduplicate by sentence text (default on)")
    parser.add_argument("--no_deduplicate", action="store_true", help="Disable deduplication")

    # Sampling knobs
    parser.add_argument("--entity_share", type=float, default=0.70, help="Share of entity-rich sentences (0..1)")
    parser.add_argument("--len_short_max", type=int, default=15, help="Max tokens for 'short' bin")
    parser.add_argument("--len_medium_max", type=int, default=30, help="Max tokens for 'medium' bin")

    args = parser.parse_args()

    require_verb = args.require_verb and not args.no_require_verb
    dedup = args.deduplicate and not args.no_deduplicate

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAMPLING SENTENCES FOR MANUAL ANNOTATION (IMPROVED)")
    print("=" * 60)

    # Load
    sentences = load_sentences(input_path, max_sentences=args.max_sentences_load)

    # Filter
    reject_stats = Counter()
    kept: List[Sentence] = []
    seen = set()

    for s in sentences:
        if not is_good_candidate(
            s,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            max_num_ratio=args.max_num_ratio,
            max_punct_ratio=args.max_punct_ratio,
            max_sym_ratio=args.max_sym_ratio,
            min_alpha_ratio=args.min_alpha_ratio,
            require_verb=require_verb,
            require_end_punct=args.require_end_punct,
            reject_stats=reject_stats,
        ):
            continue

        if dedup:
            key = normalize_text(s.get_text())
            if key in seen:
                reject_stats["deduped"] += 1
                continue
            seen.add(key)

        kept.append(s)

    print("\nFiltering summary:")
    total_seen = sum(v for k, v in reject_stats.items() if k != "kept")
    print(f"  Input sentences:     {len(sentences)}")
    print(f"  Kept candidates:     {len(kept)}")
    print(f"  Rejected (total):    {len(sentences) - len(kept)}")
    for k, v in reject_stats.most_common():
        if k == "kept":
            continue
        print(f"  - {k:18s}: {v}")

    if not kept:
        raise SystemExit("No sentences left after filtering. Relax constraints (e.g., --min_tokens, --max_num_ratio).")

    # Sample
    sampled = stratified_sample(
        kept,
        num_samples=args.num_sentences,
        seed=args.seed,
        entity_share=args.entity_share,
        len_short_max=args.len_short_max,
        len_medium_max=args.len_medium_max,
    )

    # Save
    output_file = output_dir / "sample_sentences.conllu"
    save_as_conllu(sampled, output_file)

    # Preview
    print("\nPreview (first 5 sentences):")
    for i, s in enumerate(sampled[:5], 1):
        txt = s.get_text()
        print(f"  {i}. ({s.num_tokens()} tok, entity={s.has_potential_entities()}): {txt[:120]}{'...' if len(txt)>120 else ''}")

    print(f"\nSaved {len(sampled)} sentences to: {output_file}")
    print("11th column is '_' (ready for NER tags).")
    print("Done!")


if __name__ == "__main__":
    main()

