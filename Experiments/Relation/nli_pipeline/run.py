"""R0 — NLI Pipeline: DeBERTa-v3-large zero-shot relation extraction.

Same model and verbalizations used in the main pipeline (Sainz et al. 2021
principle with Ranga & Etzkowitz 2013 verbalizations).

Difference from nli_sainz (R1):
  - Model: cross-encoder/nli-deberta-v3-large (vs roberta-large-mnli)
  - Input: central_sent_text only (vs ±1 sentence window)
  - Verbalizations: same pipeline templates

Run:
    python Experiments/Relation/nli_pipeline/run.py

Requires:
    pip install transformers torch scikit-learn
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    mark_entities,
    save_outputs,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"
NLI_THRESHOLD = 0.5


def main() -> None:
    from src.pipeline.nli_relation_extraction import NLIRelationScorer

    eval_entries = load_relation_eval()
    print(f"Loaded {len(eval_entries)} labeled relation examples")

    scorer = NLIRelationScorer(
        model_name=NLI_MODEL,
        threshold=NLI_THRESHOLD,
        batch_size=16,
    )

    true_labels, pred_labels = [], []
    nli_items = [
        (e.get("sentence") or e.get("central_sent_text") or e.get("sent_text", ""),
         e["entity_1"],
         e["entity_2"])
        for e in eval_entries
    ]

    results = scorer.score_pairs_batch(nli_items)

    for entry, result in zip(eval_entries, results):
        true_labels.append(entry["true_relation"])
        pred_labels.append(result["relation_type"])

    predictions = [
        {
            "id":   i,
            "true": t,
            "pred": p,
            "text": eval_entries[i].get("sentence") or eval_entries[i].get("central_sent_text", ""),
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
