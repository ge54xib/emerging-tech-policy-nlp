"""Shared utilities for the Experiment Suite.

Provides:
    load_relation_eval()  — load annotation.json for relation experiments
    load_spaces_eval()    — load annotation.json (dedup by sentence) for space experiments
    save_outputs()        — save predictions.json + metrics.json to outputs/

Both loaders read from evaluation/annotation.json (combined annotation file).
Each entry has: doc_id, country, sentence, entity_1, h1, entity_2, h2,
                entities, true_relation, true_space

Output format (predictions.json):
    [{"id": int, "true": str, "pred": str, "text": str}]

Output format (metrics.json):
    {"accuracy": float, "macro_f1": float, "weighted_f1": float,
     "per_class": {label: {"precision": f, "recall": f, "f1": f, "support": n}}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
ANNOTATION_FILE = _REPO_ROOT / "evaluation" / "annotation.json"
# Legacy alias kept for backwards compat
ANNOTATION_RELATION = ANNOTATION_FILE
ANNOTATION_SPACES   = ANNOTATION_FILE

RELATION_LABELS = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
    "no_explicit_relation",
]

SPACE_LABELS = [
    "knowledge_space",
    "innovation_space",
    "consensus_space",
    "public_space",
    "no_explicit_space",
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_relation_eval(path: str | Path = ANNOTATION_FILE) -> list[dict]:
    """Load annotation.json. Returns only entries with true_relation filled in.

    Each entry has keys:
        doc_id, country, sentence, entity_1, h1, entity_2, h2, entities,
        true_relation, true_space
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    labeled = [e for e in entries if e.get("true_relation", "").strip()]
    if not labeled:
        raise ValueError(
            f"No labeled entries found in {path}. "
            "Fill in 'true_relation' for each entry first."
        )
    return labeled


def load_spaces_eval(path: str | Path = ANNOTATION_FILE) -> list[dict]:
    """Load annotation.json and deduplicate by sentence for space evaluation.

    Returns only entries with true_space filled in.
    Each entry has keys:
        doc_id, country, sentence, entities, true_space
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    labeled = [e for e in entries if e.get("true_space", "").strip()]
    if not labeled:
        raise ValueError(
            f"No labeled entries found in {path}. "
            "Fill in 'true_space' for each entry first."
        )
    # Deduplicate by sentence — space is a sentence-level property
    seen: set[str] = set()
    unique: list[dict] = []
    for e in labeled:
        sent = e.get("sentence", "")
        if sent and sent not in seen:
            seen.add(sent)
            unique.append(e)
    return unique


# ── Entity marking ────────────────────────────────────────────────────────────

def mark_entities(text: str, e1: str, e2: str) -> str:
    """Insert [E1]/[/E1] and [E2]/[/E2] markers around entity spans.

    Uses case-insensitive search. If either entity is not found in the text,
    returns the text unchanged (entity may be abbreviated or lowercased).
    """
    import re

    def _insert(t: str, entity: str, open_tag: str, close_tag: str) -> str:
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        m = pattern.search(t)
        if m:
            return t[: m.start()] + open_tag + t[m.start(): m.end()] + close_tag + t[m.end():]
        return t

    marked = _insert(text, e1, "[E1] ", " [/E1]")
    marked = _insert(marked, e2, "[E2] ", " [/E2]")
    return marked


def mark_entities_typed(text: str, e1: str, h1: str, e2: str, h2: str) -> str:
    """Insert typed entity markers for LLM prompts: <helix>entity</helix>.

    Per GPT-RE convention (Wan et al. 2023) — helix label as entity type tag.
    """
    import re

    def _insert(t: str, entity: str, open_tag: str, close_tag: str) -> str:
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        m = pattern.search(t)
        if m:
            return t[: m.start()] + open_tag + t[m.start(): m.end()] + close_tag + t[m.end():]
        return t

    marked = _insert(text, e1, f"<{h1}>", f"</{h1}>")
    marked = _insert(marked, e2, f"<{h2}>", f"</{h2}>")
    return marked


# ── Output saving ─────────────────────────────────────────────────────────────

def save_outputs(
    approach_dir: str | Path,
    predictions: list[dict[str, Any]],
    true_labels: list[str],
    pred_labels: list[str],
    label_set: list[str] | None = None,
) -> None:
    """Save predictions.json and metrics.json to <approach_dir>/outputs/.

    Args:
        approach_dir: path to the approach folder (e.g. Experiments/Relation/nli_sainz)
        predictions:  list of {id, true, pred, text} dicts
        true_labels:  ground-truth labels (same order as predictions)
        pred_labels:  predicted labels (same order as predictions)
        label_set:    ordered list of valid labels (for consistent per_class keys)
    """
    out_dir = Path(approach_dir) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # predictions.json
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # metrics.json
    report = classification_report(
        true_labels, pred_labels,
        labels=label_set,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"], 4),
            "f1":        round(report[lbl]["f1-score"], 4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in (label_set or [])
        if lbl in report
    }
    metrics = {
        "accuracy":    round(accuracy_score(true_labels, pred_labels), 4),
        "macro_f1":    round(f1_score(true_labels, pred_labels, average="macro",
                                      labels=label_set, zero_division=0), 4),
        "weighted_f1": round(f1_score(true_labels, pred_labels, average="weighted",
                                      labels=label_set, zero_division=0), 4),
        "per_class":   per_class,
        "n_eval":      len(true_labels),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Console summary
    _print_table(metrics, approach_dir)


def _print_table(metrics: dict, approach_dir: str | Path) -> None:
    name = Path(approach_dir).name
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  N eval:      {metrics['n_eval']}")
    print(f"\n  {'Label':<40} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-'*62}")
    for lbl, v in metrics["per_class"].items():
        print(f"  {lbl:<40} {v['precision']:>6.3f} {v['recall']:>6.3f} "
              f"{v['f1']:>6.3f} {v['support']:>5}")
    print()
