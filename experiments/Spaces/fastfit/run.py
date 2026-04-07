"""S4 — FastFit: fast and effective few-shot classification for similar classes.

Paper: Yehudai, A. & Bandel, E. (2024). FastFit: Fast and Effective Few-Shot
       Text Classification with a Multitude of Classes.
       NAACL 2024 System Demonstrations.

Method (from paper):
  - Key contribution: token-level similarity instead of sentence-level contrastive loss
  - Similarity score = sum of max cosine similarities between query tokens and support tokens
    (batch token-level attention across all support examples simultaneously)
  - Loss: batch contrastive learning across ALL support examples at once
    (unlike SetFit's pairwise sampling — more efficient, more accurate for similar classes)
  - 3-20x faster training than SetFit; designed for tasks with many semantically similar classes
  - Directly addresses knowledge_space / consensus_space overlap in this dataset

Training data: spaces_labels.json (25 per class = 100 total)
               optionally + spaces_llm_review.json

Run:
    pip install fast-fit
    python Experiments/Spaces/fastfit/run.py

Requires:
    pip install fast-fit scikit-learn datasets
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

BASE_MODEL         = "roberta-base"
NUM_TRAIN_EPOCHS   = 10    # paper recommendation for few-shot regime
NUM_ITERATIONS     = 5     # number of repetitions per batch (paper default)
LEARNING_RATE      = 3e-5
BATCH_SIZE         = 32


def _load_training_data() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for path in [REVIEW_FILE, LLM_FILE]:
        if not path.exists():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        for e in entries:
            space    = e.get("space") or e.get("true_space", "")
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
    from fastfit import FastFitTrainer

    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    train_texts, train_labels = _load_training_data()


    train_ds = Dataset.from_dict({
        "text":  train_texts,
        "label": train_labels,
    })

    eval_texts = [e["sentence"] for e in eval_entries]
    eval_true  = [e["true_space"] for e in eval_entries]
    eval_ds    = Dataset.from_dict({
        "text":  eval_texts,
        "label": eval_true,
    })

    trainer = FastFitTrainer(
        model_name_or_path=BASE_MODEL,
        label_column_name="label",
        text_column_name="text",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        train_dataset=train_ds,
        test_dataset=eval_ds,
    )

    model = trainer.train()

    # Direct model inference (bypass HF Trainer collator which fails on string labels)
    import numpy as np
    import torch
    id2label  = model.config.id2label
    tokenizer = trainer.tokenizer
    model.eval()
    device = next(model.parameters()).device
    all_logits = []
    for i in range(0, len(eval_texts), 32):
        batch = tokenizer(
            eval_texts[i:i+32],
            padding=True, truncation=True, max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**batch)
        logits = out.logits if hasattr(out, "logits") else out[0]
        all_logits.append(logits.cpu().numpy())
    pred_indices = np.concatenate(all_logits, axis=0).argmax(axis=-1)
    pred_labels  = [id2label.get(int(i), str(i)) for i in pred_indices]

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_texts[i]}
        for i, (t, p) in enumerate(zip(eval_true, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, eval_true, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
