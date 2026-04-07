"""Aggregate all experiment results and print a comparison table.

Scans:
  Experiments/Relation/*/outputs/metrics.json
  Experiments/Spaces/*/outputs/metrics.json

Prints formatted tables sorted by macro_f1 descending.
Saves Experiments/RESULTS.md with markdown tables.

Run:
    python Experiments/compare_all.py
"""
from __future__ import annotations

import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_EXPERIMENTS_DIR = Path(__file__).parent

RELATION_LABELS = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
    "no_explicit_relation",
]
RELATION_SHORT = {
    "technology_transfer":              "tech_tr",
    "collaboration_conflict_moderation":"collab_conf",
    "collaborative_leadership":         "collab_lead",
    "substitution":                     "subst",
    "networking":                       "network",
    "no_explicit_relation":             "no_rel",
}

SPACE_LABELS = [
    "knowledge_space",
    "innovation_space",
    "consensus_space",
    "public_space",
    "no_explicit_space",
]
SPACE_SHORT = {
    "knowledge_space":  "know",
    "innovation_space": "innov",
    "consensus_space":  "consens",
    "public_space":     "public",
    "no_explicit_space": "none",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_metrics(metrics_path: Path) -> dict | None:
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect(task_dir: Path, task_labels: list[str]) -> list[dict]:
    """Return list of result rows for a given task directory."""
    rows = []
    for approach_dir in sorted(task_dir.iterdir()):
        if not approach_dir.is_dir():
            continue
        metrics_file = approach_dir / "outputs" / "metrics.json"
        if not metrics_file.exists():
            continue
        m = _load_metrics(metrics_file)
        if m is None:
            continue
        row = {
            "approach":     approach_dir.name,
            "macro_f1":     m.get("macro_f1", 0.0),
            "weighted_f1":  m.get("weighted_f1", 0.0),
            "accuracy":     m.get("accuracy", 0.0),
            "per_class":    m.get("per_class", {}),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


def _fmt(val: float | None) -> str:
    if val is None:
        return "  —  "
    return f"{val:.3f}"


def _best(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else -1.0


def _best_per_class(rows: list[dict], label: str) -> float:
    vals = [
        r["per_class"].get(label, {}).get("f1", None)
        for r in rows
        if r["per_class"].get(label, {}).get("f1") is not None
    ]
    return max(vals) if vals else -1.0


# ── Console table ─────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list[dict], labels: list[str], short: dict[str, str]) -> None:
    if not rows:
        print(f"\n{title}: no results found.\n")
        return

    col_labels = [short[l] for l in labels]
    header = f"{'Approach':<28} {'MacroF1':>7} {'WtdF1':>7} {'Acc':>7}  " + \
             "  ".join(f"{c:>8}" for c in col_labels)
    sep = "-" * len(header)

    best_macro   = _best(rows, "macro_f1")
    best_wtd     = _best(rows, "weighted_f1")
    best_acc     = _best(rows, "accuracy")
    best_pc      = {l: _best_per_class(rows, l) for l in labels}

    print(f"\n{'=' * len(header)}")
    print(f"  {title}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for r in rows:
        macro  = r["macro_f1"]
        wtd    = r["weighted_f1"]
        acc    = r["accuracy"]

        def _star(val: float, best: float) -> str:
            s = _fmt(val)
            return f"*{s}*" if abs(val - best) < 1e-6 else f" {s} "

        pc_cols = []
        for l in labels:
            f1 = r["per_class"].get(l, {}).get("f1", None)
            if f1 is None:
                pc_cols.append("    —   ")
            else:
                marker = "*" if abs(f1 - best_pc[l]) < 1e-6 else " "
                pc_cols.append(f"{marker}{_fmt(f1)}{marker}")

        print(
            f"{r['approach']:<28} "
            f"{_star(macro, best_macro):>9} "
            f"{_star(wtd,   best_wtd):>9} "
            f"{_star(acc,   best_acc):>9}  "
            + "  ".join(c for c in pc_cols)
        )
    print(sep)
    print("  * = best in column")


# ── Markdown table ────────────────────────────────────────────────────────────

def _md_table(title: str, rows: list[dict], labels: list[str], short: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if not rows:
        lines.append(f"### {title}\n\n_No results yet._\n")
        return lines

    col_labels = [short[l] for l in labels]
    header_cols = ["Approach", "Macro F1", "Wtd F1", "Acc"] + col_labels
    lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    best_macro = _best(rows, "macro_f1")
    best_pc    = {l: _best_per_class(rows, l) for l in labels}

    for r in rows:
        macro  = r["macro_f1"]
        wtd    = r["weighted_f1"]
        acc    = r["accuracy"]

        def _md_val(val: float, best: float) -> str:
            s = f"{val:.3f}"
            return f"**{s}**" if abs(val - best) < 1e-6 else s

        pc_cells = []
        for l in labels:
            f1 = r["per_class"].get(l, {}).get("f1", None)
            if f1 is None:
                pc_cells.append("—")
            else:
                pc_cells.append(_md_val(f1, best_pc[l]))

        row_cells = [
            r["approach"],
            _md_val(macro, best_macro),
            f"{wtd:.3f}",
            f"{acc:.3f}",
        ] + pc_cells

        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    relation_dir = _EXPERIMENTS_DIR / "Relation"
    spaces_dir   = _EXPERIMENTS_DIR / "Spaces"

    rel_rows   = _collect(relation_dir, RELATION_LABELS) if relation_dir.exists() else []
    space_rows = _collect(spaces_dir,   SPACE_LABELS)   if spaces_dir.exists()   else []

    # Console output
    _print_table("Relation Classification", rel_rows,   RELATION_LABELS, RELATION_SHORT)
    _print_table("Space Classification",    space_rows, SPACE_LABELS,    SPACE_SHORT)

    # Markdown output
    md_lines: list[str] = [
        "# Experiment Results\n",
        f"_Generated by `Experiments/compare_all.py`_\n",
        "",
    ]
    md_lines += _md_table("Relation Classification", rel_rows,   RELATION_LABELS, RELATION_SHORT)
    md_lines += _md_table("Space Classification",    space_rows, SPACE_LABELS,    SPACE_SHORT)

    results_path = _EXPERIMENTS_DIR / "RESULTS.md"
    results_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved → {results_path}")


if __name__ == "__main__":
    main()
