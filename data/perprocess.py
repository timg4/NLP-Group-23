#!/usr/bin/env python
"""
Preprocess FinCorpus-DE10k for Milestone 1 (NLP & IE 2025WS)
-------------------------------------------------------------
Performs:
 - light normalization (whitespace cleanup, abbreviation protection)
 - sentence segmentation, tokenization, POS tagging, lemmatization
 - export of all processed documents into ONE CoNLL-U file

Usage example:
    python src/preprocess_fincorpus.py \
        --output data/processed/fincorpus_processed.conllu \
        --max_docs 500
"""

from pydoc import doc
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import stanza
from stanza.utils.conll import CoNLL
from datasets import load_dataset

# ---------------------------------------------------------------------
# Abbreviation protection for German legal/financial domain
# ---------------------------------------------------------------------
ABBREV = [
    r"Abs\.", r"Nr\.", r"Art\.", r"Kap\.", r"Anm\.",
    r"z\. ?B\.", r"u\. ?a\.", r"vgl\.", r"ca\.", r"bzw\.",
    r"Dr\.", r"Dipl\.", r"Prof\.", r"Hr\.", r"Fr\.",
    r"i\. ?V\.", r"i\. ?S\.", r"i\. ?d\. ?R\."
]
ABBREV_RX = re.compile(r"(" + "|".join(ABBREV) + r")")

def protect_abbrev(text: str) -> str:
    """Replace '.' in known abbreviations with a placeholder."""
    return ABBREV_RX.sub(lambda m: m.group(0).replace(".", "⟂"), text)

def normalize_paragraph_sign(text: str) -> str:
    """Ensure consistent spacing around § symbol."""
    return re.sub(r"§\s+", "§ ", text)

def light_clean(text: str) -> str:
    """Apply lightweight normalization before stanza processing."""
    t = text.strip()
    t = normalize_paragraph_sign(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = protect_abbrev(t)
    return t

# ---------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------
def preprocess(output_path: Path, max_docs: int | None = None):
    print("Loading FinCorpus-DE10k dataset...")
    dataset = load_dataset("data/fincorpus-de-10k.py", split="train")
    print(f"Dataset loaded: {len(dataset)} documents")

    print("Loading Stanza German pipeline...")
    nlp = stanza.Pipeline(
        "de",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing combined CoNLL-U file to {output_path}\n")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(dataset, desc="Documents")):
            if max_docs and i >= max_docs:
                break

            text = sample.get("text", "").strip()
            if not text:
                continue

            text_norm = light_clean(text)
            doc = nlp(text_norm)

            fout.write(f"# newdoc id = {i}\n")
            preview = text_norm[:200].replace("\n", " ")
            fout.write(f"# text = {preview}...\n")

            conllu_blocks = CoNLL.convert_dict(doc.to_dict())

            flat_lines = []
            for block in conllu_blocks:
                if isinstance(block, list):
                    for line in block:
                        if isinstance(line, list):
                            flat_lines.extend(line)
                        else:
                            flat_lines.append(line)
                else:
                    flat_lines.append(block)


            fout.write("\n".join(flat_lines) + "\n")

# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FinCorpus-DE10k to one CoNLL-U file.")
    parser.add_argument("--output", required=True, help="Path to combined .conllu output file.")
    parser.add_argument("--max_docs", type=int, default=None, help="Limit number of documents to process.")
    args = parser.parse_args()

    preprocess(Path(args.output), args.max_docs)
