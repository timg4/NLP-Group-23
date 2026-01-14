"""
Shared utilities for final submission runs.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import classification_report

from utils.stratified_split import SPLIT_SEED, stratified_split
from data_loader import _align_tokens, _tokens_to_spans

LABELS = ["ORG", "MON", "LEG"]
LABEL_ORDER = ["LEG", "MON", "ORG", "O"]


def _reconstruct_text(tokens: List[Dict]) -> str:
    parts = []
    for tok in tokens:
        parts.append(tok["form"])
        if "SpaceAfter=No" not in tok.get("misc", ""):
            parts.append(" ")
    return "".join(parts).strip()


def load_conllu_full(path: str) -> List[Dict]:
    """
    Load CoNLL-U with tokens, labels, and sentence text.
    """
    sentences = []
    tokens: List[Dict] = []
    labels: List[str] = []
    sent_text = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    text = sent_text or _reconstruct_text(tokens)
                    token_forms = [t["form"] for t in tokens]
                    spans = _tokens_to_spans(list(zip(token_forms, labels)), text)
                    sentences.append(
                        {
                            "text": text,
                            "tokens": tokens,
                            "token_forms": token_forms,
                            "labels": labels,
                            "spans": spans,
                        }
                    )
                tokens, labels, sent_text = [], [], None
                continue

            if line.startswith("# text ="):
                sent_text = line.split("=", 1)[1].strip()
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 11 or "-" in fields[0]:
                continue

            token = {
                "form": fields[1],
                "upos": fields[3],
                "misc": fields[9],
            }
            tokens.append(token)
            labels.append(fields[10].strip())

    if tokens:
        text = sent_text or _reconstruct_text(tokens)
        token_forms = [t["form"] for t in tokens]
        spans = _tokens_to_spans(list(zip(token_forms, labels)), text)
        sentences.append(
            {
                "text": text,
                "tokens": tokens,
                "token_forms": token_forms,
                "labels": labels,
                "spans": spans,
            }
        )

    return sentences


def build_data_by_label(items: List[Dict]) -> Dict[str, List]:
    data_by_label = {lab: [] for lab in LABELS}
    for item in items:
        text = item["text"]
        spans = item["spans"]
        for label in LABELS:
            label_spans = [
                {"start": sp["start"], "end": sp["end"], "text": sp["text"]}
                for sp in spans
                if sp["label"] == label
            ]
            data_by_label[label].append((text, label_spans))
    return data_by_label


def spans_to_bio(spans: List[Dict], tokens: List[str], text: str) -> List[str]:
    aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        text = " ".join(tokens)
        aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        return ["O"] * len(tokens)

    offsets = [(start, end) for _, start, end, _ in aligned]
    labels = ["O"] * len(tokens)

    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        label = span["label"]
        s_start = span["start"]
        s_end = span["end"]
        in_span = False
        for i, (t_start, t_end) in enumerate(offsets):
            if t_end <= s_start or t_start >= s_end:
                if in_span:
                    in_span = False
                continue
            if labels[i] != "O":
                continue
            labels[i] = f"B-{label}" if not in_span else f"I-{label}"
            in_span = True

    return labels


def collapse_bio(labels: List[str]) -> List[str]:
    collapsed = []
    for lab in labels:
        if lab.startswith("B-") or lab.startswith("I-"):
            collapsed.append(lab.split("-", 1)[1])
        else:
            collapsed.append("O")
    return collapsed


def evaluate_predictions(
    gold_labels: List[List[str]], pred_labels: List[List[str]]
) -> Tuple[str, Dict]:
    y_true = [lab for sent in gold_labels for lab in sent]
    y_pred = [lab for sent in pred_labels for lab in sent]
    report_text = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    return report_text, report_dict


def labels_to_spans(labels: List[str], tokens: List[str], text: str) -> List[Dict]:
    aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        text = " ".join(tokens)
        aligned = _align_tokens([(t, "O") for t in tokens], text)
    if len(aligned) != len(tokens):
        return []

    spans = []
    cur_label = None
    cur_start = None
    cur_end = None

    def _flush():
        if cur_label is not None and cur_start is not None and cur_end is not None:
            spans.append(
                {
                    "label": cur_label,
                    "start": cur_start,
                    "end": cur_end,
                    "text": text[cur_start:cur_end],
                }
            )

    for (_, start, end, _), lab in zip(aligned, labels):
        if lab.startswith("B-") or lab.startswith("I-"):
            lab_norm = lab.split("-", 1)[1]
        else:
            lab_norm = lab

        if lab_norm == "O":
            _flush()
            cur_label = cur_start = cur_end = None
            continue

        if cur_label is None or cur_label != lab_norm:
            _flush()
            cur_label = lab_norm
            cur_start = start
            cur_end = end
        else:
            cur_end = end

    _flush()
    return spans


def evaluate_overlap(
    gold_spans_all: List[List[Dict]],
    pred_spans_all: List[List[Dict]],
) -> Dict[str, Dict[str, float]]:
    tp = {lab: 0 for lab in LABELS}
    fp = {lab: 0 for lab in LABELS}
    fn = {lab: 0 for lab in LABELS}

    for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
        gold_by_label = {lab: [] for lab in LABELS}
        pred_by_label = {lab: [] for lab in LABELS}

        for sp in gold_spans:
            if sp["label"] in gold_by_label:
                gold_by_label[sp["label"]].append(sp)
        for sp in pred_spans:
            if sp["label"] in pred_by_label:
                pred_by_label[sp["label"]].append(sp)

        for lab in LABELS:
            golds = gold_by_label[lab]
            preds = pred_by_label[lab]
            matched = set()

            for pred in preds:
                hit = False
                for i, gold in enumerate(golds):
                    if i in matched:
                        continue
                    if not (pred["end"] <= gold["start"] or pred["start"] >= gold["end"]):
                        tp[lab] += 1
                        matched.add(i)
                        hit = True
                        break
                if not hit:
                    fp[lab] += 1
            fn[lab] += len(golds) - len(matched)

    results = {}
    for lab in LABELS:
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
        results[lab] = {"precision": p, "recall": r, "f1": f1}

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    results["micro"] = {"precision": p, "recall": r, "f1": f1}

    return results


def save_predictions_conllu(sentence_predictions: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sent_pred in enumerate(sentence_predictions):
            f.write(f"# sent_id = {i + 1}\n")
            for j, (token, label) in enumerate(
                zip(sent_pred["tokens"], sent_pred["pred"])
            ):
                f.write(f"{j+1}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\t{label}\n")
            f.write("\n")


def prepare_split(data_path: str):
    all_sents = load_conllu_full(data_path)
    train_set, dev_set, split_counts = stratified_split(
        all_sents, train_ratio=0.8, seed=SPLIT_SEED
    )
    return train_set, dev_set, split_counts


def print_split_counts(split_counts: Dict[str, Dict[str, int]]):
    print("Stratified split by sentence label:")
    for key in sorted(split_counts.keys()):
        c = split_counts[key]
        print(f"  {key}: total={c['total']}, train={c['train']}, dev={c['dev']}")


def write_metrics(
    out_dir: Path,
    method_name: str,
    report_text: str,
    report_dict: Dict,
    elapsed_s: float,
    overlap_metrics: Dict[str, Dict[str, float]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_summary.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{method_name} METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Timing (seconds)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{method_name}: {elapsed_s:.2f}\n\n")
        f.write(report_text + "\n")
        if overlap_metrics:
            f.write("\nOverlap metrics (span-level)\n")
            f.write("-" * 80 + "\n")
            for lab in LABELS + ["micro"]:
                m = overlap_metrics.get(lab, {})
                f.write(
                    f"{lab}: P={m.get('precision', 0.0):.3f} "
                    f"R={m.get('recall', 0.0):.3f} "
                    f"F1={m.get('f1', 0.0):.3f}\n"
                )
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "method": "{method_name}",\n')
        f.write(f'  "seconds": {elapsed_s:.6f},\n')
        f.write('  "report": ')
        f.write(_dict_to_json(report_dict, indent=2))
        if overlap_metrics:
            f.write(",\n")
            f.write('  "overlap": ')
            f.write(_dict_to_json(overlap_metrics, indent=2))
        f.write("\n}\n")


def _dict_to_json(data, indent: int = 2) -> str:
    import json

    return json.dumps(data, indent=indent, ensure_ascii=True)
