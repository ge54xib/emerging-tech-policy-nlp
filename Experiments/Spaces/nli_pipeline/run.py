"""S0 — NLI Pipeline: DeBERTa-v3-large zero-shot TH space classification.

Runs NLI classify_spaces_batch() directly on eval sentences.
No precomputed pipeline output required.

Model: cross-encoder/nli-deberta-v3-large
Space hypotheses: "This text is about {label}." for each of 4 TH spaces.

Run:
    python Experiments/Spaces/nli_pipeline/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

NLI_MODEL     = "cross-encoder/nli-deberta-v3-large"
NLI_THRESHOLD = 0.5


def main() -> None:
    from src.pipeline.nli_relation_extraction import NLIRelationScorer

    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    scorer = NLIRelationScorer(
        model_name=NLI_MODEL,
        threshold=NLI_THRESHOLD,
        batch_size=16,
    )

    sentences = [entry["sentence"] for entry in eval_entries]
    results = scorer.classify_spaces_batch(sentences)

    true_labels, pred_labels = [], []
    for entry, result in zip(eval_entries, results):
        true_labels.append(entry["true_space"])
        pred_labels.append(result["th_space"])

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_entries[i]["sentence"]}
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
