#!/usr/bin/env python3
"""
Summarize per-model metrics into a single file.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=Path(__file__).parent.parent / "results",
        help="Base results directory containing per-model folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    models = []
    skip_methods = {"TokenNB_data_priors"}

    for metrics_file in results_dir.rglob("metrics.json"):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        method = data.get("method", metrics_file.parent.name)
        if method in skip_methods:
            continue
        report = data.get("report", {})
        models.append(
            {
                "method": method,
                "seconds": data.get("seconds", 0.0),
                "LEG_f1": report.get("LEG", {}).get("f1-score", 0.0),
                "MON_f1": report.get("MON", {}).get("f1-score", 0.0),
                "ORG_f1": report.get("ORG", {}).get("f1-score", 0.0),
                "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
                "LEG_f1_overlap": data.get("overlap", {}).get("LEG", {}).get("f1", 0.0),
                "MON_f1_overlap": data.get("overlap", {}).get("MON", {}).get("f1", 0.0),
                "ORG_f1_overlap": data.get("overlap", {}).get("ORG", {}).get("f1", 0.0),
                "accuracy": report.get("accuracy", 0.0),
            }
        )

    models.sort(key=lambda x: x["method"])

    out_path = results_dir / "summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUBMISSION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        header = (
            f"{'Method':25} {'LEG_F1':>8} {'MON_F1':>8} {'ORG_F1':>8} "
            f"{'LEG_F1~':>8} {'MON_F1~':>8} {'ORG_F1~':>8} "
            f"{'Macro_F1':>9} {'Acc':>8} {'Sec':>8}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")
        for m in models:
            f.write(
                f"{m['method'][:25]:25} "
                f"{m['LEG_f1']:8.3f} "
                f"{m['MON_f1']:8.3f} "
                f"{m['ORG_f1']:8.3f} "
                f"{m['LEG_f1_overlap']:8.3f} "
                f"{m['MON_f1_overlap']:8.3f} "
                f"{m['ORG_f1_overlap']:8.3f} "
                f"{m['macro_f1']:9.3f} "
                f"{m['accuracy']:8.3f} "
                f"{m['seconds']:8.2f}\n"
            )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
