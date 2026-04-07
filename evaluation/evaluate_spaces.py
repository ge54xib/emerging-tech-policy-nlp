"""Compute F1 score for SetFit TH-space predictions against manual annotations.

Run after filling in annotation_spaces.json:
    python evaluation/evaluate_spaces.py
"""

import json
from collections import defaultdict
from pathlib import Path

ANNOTATION_FILE = Path(__file__).parent / "annotation_spaces.json"

SPACE_LABELS = ["knowledge_space", "innovation_space", "consensus_space", "public_space"]


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def main():
    entries = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))

    annotated = [e for e in entries if str(e.get("true_space", "")).strip()
                 and str(e.get("setfit_space", "")).strip()]
    if not annotated:
        print("No annotated entries found. Fill in 'true_space' fields first.")
        return

    print(f"Evaluating {len(annotated)} annotated entries...\n")

    y_true = [e["true_space"].strip() for e in annotated]
    y_pred = [e["setfit_space"].strip() for e in annotated]

    tp_by_class = defaultdict(int)
    fp_by_class = defaultdict(int)
    fn_by_class = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp_by_class[true] += 1
        else:
            fp_by_class[pred] += 1
            fn_by_class[true] += 1

    print(f"{'Class':<35} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print("-" * 65)

    all_labels = sorted(set(y_true) | set(y_pred))
    support = defaultdict(int)
    for t in y_true:
        support[t] += 1

    macro_f1s = []
    weighted_f1s = []

    for label in all_labels:
        p, r, f1 = precision_recall_f1(tp_by_class[label], fp_by_class[label], fn_by_class[label])
        n = support[label]
        print(f"  {label:<33} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {n:>8}")
        macro_f1s.append(f1)
        weighted_f1s.append(f1 * n)

    total = len(y_true)
    macro = round(sum(macro_f1s) / len(macro_f1s), 3) if macro_f1s else 0
    weighted = round(sum(weighted_f1s) / total, 3) if total > 0 else 0
    accuracy = round(sum(t == p for t, p in zip(y_true, y_pred)) / total, 3)

    print("-" * 65)
    print(f"  {'Macro F1':<33} {'':>6} {'':>6} {macro:>6.3f} {total:>8}")
    print(f"  {'Weighted F1':<33} {'':>6} {'':>6} {weighted:>6.3f} {total:>8}")
    print(f"  {'Accuracy':<33} {'':>6} {'':>6} {accuracy:>6.3f} {total:>8}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    labels_present = sorted(set(y_true))
    header = f"{'':>30}" + "".join(f"{l[:8]:>12}" for l in labels_present)
    print(header)
    for true_label in labels_present:
        row_str = f"  {true_label:<28}"
        for pred_label in labels_present:
            count = sum(1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label)
            row_str += f"{count:>12}"
        print(row_str)


if __name__ == "__main__":
    main()
