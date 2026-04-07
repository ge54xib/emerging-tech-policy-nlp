"""R1 — NLI-based zero-shot relation extraction with label verbalization.

Paper: Sainz, O., et al. (2021). Label Verbalization and Entailment for
       Effective Zero and Few-Shot Relation Extraction. EMNLP 2021.

Method (from paper):
  - Entity-marked sentence as NLI premise
  - One verbalized hypothesis per relation: "[e1] [relation description] [e2]."
  - Score = entailment probability; argmax → predicted label
  - Single direction (e1 → e2), no bidirectional averaging
  - No threshold — always pick highest-scoring label

Entity marking: [E1] entity_1 [/E1] ... [E2] entity_2 [/E2]
  inserted into central_sent_text via case-insensitive string search.

Verbalizations grounded in Ranga & Etzkowitz (2013):
  - technology_transfer: "market or non-market transfer of technology..."
  - collaboration_conflict_moderation: "turned tension and conflict of interest..."
  - collaborative_leadership: "exercised convening power..."
  - substitution: "filled a gap that emerged because ... was weak..."
  - networking: "formed a formal or informal network..."
  - no_explicit_relation: fallback

Run:
    python Experiments/Relation/nli_sainz/run.py

Requires:
    pip install transformers torch scikit-learn
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make shared/ importable from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    mark_entities,
    save_outputs,
)

# ── Verbalization templates (Sainz et al. 2021 format, R&E 2013 wording) ──────
# Format: single factual statement; argmax over entailment probabilities.
VERBALIZATIONS: dict[str, str] = {
    "technology_transfer": (
        "{e1} transferred technology to {e2} via market or non-market interactions."
    ),
    "collaboration_conflict_moderation": (
        "{e1} turned tension and conflict of interest with {e2} "
        "into convergence and confluence of interest."
    ),
    "collaborative_leadership": (
        "{e1} exercised convening power to bring together {e2} and coordinated "
        "a mix of top-down and bottom-up processes."
    ),
    "substitution": (
        "{e1} filled a gap that emerged because {e2} was weak, "
        "taking the role of the other."
    ),
    "networking": (
        "{e1} formed a formal or informal network with {e2} as a manifestation "
        "of the collective nature of science, technology and innovation."
    ),
    "no_explicit_relation": (
        "{e1} and {e2} have no explicit Triple Helix relationship in this sentence."
    ),
}

MODEL_NAME = "roberta-large-mnli"   # paper-specified model


def predict(entries: list[dict]) -> tuple[list[str], list[str]]:
    from transformers import pipeline

    import torch
    _device = 0 if torch.cuda.is_available() else -1
    nli = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=_device,
        multi_label=False,
    )

    true_labels, pred_labels = [], []

    for entry in entries:
        e1   = entry["entity_1"]
        e2   = entry["entity_2"]
        text = entry.get("sentence") or entry.get("central_sent_text", "")

        # Insert entity markers into the premise
        premise = mark_entities(text, e1, e2)

        # Build one hypothesis per relation
        hypotheses = [
            VERBALIZATIONS[rel].format(e1=e1, e2=e2)
            for rel in RELATION_LABELS
        ]

        # Score all hypotheses simultaneously (single NLI call)
        result = nli(
            premise,
            candidate_labels=hypotheses,
            hypothesis_template="{}",   # hypotheses already complete sentences
        )

        # Map back from hypothesis string → label
        hyp_to_label = {
            VERBALIZATIONS[rel].format(e1=e1, e2=e2): rel
            for rel in RELATION_LABELS
        }
        best_hyp = result["labels"][0]   # highest-scoring hypothesis
        pred = hyp_to_label.get(best_hyp, "no_explicit_relation")

        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)

    return true_labels, pred_labels


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples")
    print(f"Model: {MODEL_NAME}")

    true_labels, pred_labels = predict(entries)

    predictions = [
        {
            "id":   i,
            "true": t,
            "pred": p,
            "text": entries[i].get("sentence") or entries[i].get("central_sent_text", ""),
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
