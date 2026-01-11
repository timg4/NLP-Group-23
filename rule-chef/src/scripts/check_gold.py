#this checks whether the gold data is well-formed: no overlapping spans, valid labels, etc.
from collections import Counter, defaultdict
from src.data.conllu_reader import read_conllu
from src.data.gold_to_spans import align_tokens_to_text, bio_to_spans

def main():
    path = "././gold_data/gold.conllu"
    bad = 0
    n_sent = 0
    span_counts = Counter()

    for sent in read_conllu(path):
        n_sent += 1
        text = sent.text
        char_tokens = align_tokens_to_text(sent)
        spans = bio_to_spans(char_tokens)

        # attach text + basic checks
        for sp in spans:
            sp_text = text[sp["start"]:sp["end"]]
            sp["text"] = sp_text

            assert sp["text"] == sp_text
            assert 0 <= sp["start"] < sp["end"] <= len(text)
            assert sp["label"] in {"ORG", "MON", "LEG"}

            span_counts[sp["label"]] += 1

        # overlap check per label
        by_label = defaultdict(list)
        for sp in spans:
            by_label[sp["label"]].append((sp["start"], sp["end"]))

        for lab, intervals in by_label.items():
            intervals.sort()
            for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
                if s2 < e1:
                    raise AssertionError(
                        f"Overlap in sent_id={sent.sent_id}, label={lab}: "
                        f"{(s1,e1)} overlaps {(s2,e2)}; text='{text}'"
                    )

    print(f"OK: {n_sent} sentences")
    print("Span counts:", dict(span_counts))

if __name__ == "__main__":
    main()