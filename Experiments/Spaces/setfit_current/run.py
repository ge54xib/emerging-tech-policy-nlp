"""S1 — SetFit: few-shot learning without prompts.

Paper: Tunstall, L., et al. (2022). Efficient Few-Shot Learning Without Prompts.
       EMNLP 2022.

Method (from paper):
  Stage 1 — Contrastive fine-tuning of Siamese sentence-transformer:
    - Generate positive pairs: same-label sentences
    - Generate negative pairs: different-label sentences
    - Loss: CosineSimilarityLoss on sampled pairs
    - num_iterations=20 (paper default): number of sentence pairs generated per class

  Stage 2 — Train lightweight classification head (logistic regression) on
    the fine-tuned sentence embeddings.

  Key: no prompts, no templates — pure embedding similarity and contrastive training.

Training data:
  spaces_labels.json (25 per class, human-annotated) +
  spaces_llm_review.json (GPT-4 annotated, combined)

Run:
    python Experiments/Spaces/setfit_current/run.py

Requires:
    pip install setfit scikit-learn datasets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

_REPO_ROOT    = Path(__file__).parent.parent.parent.parent
REVIEW_FILE   = Path(__file__).parent.parent / "spaces_labels.json"
LLM_FILE      = _REPO_ROOT / "data/processed/step3/spaces_llm_review.json"

SETFIT_MODEL  = "sentence-transformers/all-mpnet-base-v2"
NUM_ITERATIONS = 20   # paper default: number of positive/negative pairs per class


def _load_training_data() -> tuple[list[str], list[str]]:
    texts, labels = [], []

    for path in [REVIEW_FILE, LLM_FILE]:
        if not path.exists():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        for e in entries:
            space = e.get("space") or e.get("true_space", "")
            sentence = e.get("central_sentence") or e.get("sentence", "")
            if space in SPACE_LABELS and sentence.strip():
                texts.append(sentence.strip())
                labels.append(space)

    print(f"Training examples: {len(texts)} total")
    from collections import Counter
    for lbl, cnt in Counter(labels).items():
        print(f"  {lbl}: {cnt}")
    return texts, labels


def main() -> None:
    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer

    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    train_texts, train_labels = _load_training_data()

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    eval_texts  = [e["sentence"] for e in eval_entries]
    eval_true   = [e["true_space"] for e in eval_entries]
    eval_ds     = Dataset.from_dict({"text": eval_texts, "label": eval_true})

    model = SetFitModel.from_pretrained(SETFIT_MODEL)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric="f1",
        num_iterations=NUM_ITERATIONS,
        num_epochs=1,
    )
    trainer.train()

    pred_labels = [SPACE_LABELS[p] if isinstance(p, int) else p
                   for p in model.predict(eval_texts)]

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_texts[i]}
        for i, (t, p) in enumerate(zip(eval_true, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, eval_true, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
